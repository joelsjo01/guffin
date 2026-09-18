import copy
import json
import math
import pathlib
import random
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import numpy
    import torch
except ModuleNotFoundError:  # The offline training environment is optional for repository CI.
    numpy = None
    torch = None


@unittest.skipIf(torch is None, "offline PyTorch training dependencies are not installed")
class SpeculationTrainerOrchestrationTests(unittest.TestCase):
    def _registry(self):
        from training.speculation.registry import load_registry

        return load_registry(ROOT / "training/speculation/registry.json")

    def test_full_uint64_seed_is_not_truncated_and_numpy_derivation_uses_high_bits(self):
        from training.speculation.train import _derived_numpy_seed

        seed = 13503953896175478587
        torch.manual_seed(seed)
        self.assertEqual(torch.initial_seed(), seed)
        self.assertNotEqual(_derived_numpy_seed(7), _derived_numpy_seed(7 + (1 << 32)))
        self.assertEqual(_derived_numpy_seed(seed), _derived_numpy_seed(seed))
        for invalid in (-1, 2**64, True):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                RuntimeError, "exact uint64"
            ):
                _derived_numpy_seed(invalid)

    def test_optimizer_contract_rejects_unimplemented_or_malformed_values(self):
        from training.speculation.train import _validated_optimizer_contract

        registry = self._registry()
        self.assertIs(
            _validated_optimizer_contract(registry),
            registry.raw["default_transfer_optimizer"],
        )
        cases = {
            "algorithm": "SGD",
            "schedule": "linear",
            "precision": "f16",
            "learning_rate": math.nan,
            "betas": [0.9, 1.0],
            "minimum_epochs": 7,
            "micro_batch": 8,
            "gradient_accumulation": 0,
        }
        for field, value in cases.items():
            with self.subTest(field=field):
                raw = copy.deepcopy(registry.raw)
                raw["default_transfer_optimizer"][field] = value
                altered = SimpleNamespace(raw=raw)
                with self.assertRaises(RuntimeError):
                    _validated_optimizer_contract(altered)

    def test_root_only_training_physically_packs_one_logical_accumulation_window(self):
        from training.speculation.train import _physical_training_batch_size

        contract = self._registry().raw["default_transfer_optimizer"]
        self.assertEqual(contract["micro_batch"], 1)
        self.assertEqual(contract["gradient_accumulation"], 16)
        self.assertEqual(_physical_training_batch_size("root_only", contract), 16)
        self.assertEqual(_physical_training_batch_size("full_context", contract), 1)
        with self.assertRaisesRegex(RuntimeError, "unknown feature context mode"):
            _physical_training_batch_size("ambiguous", contract)

    def test_active_root_only_packed_loss_matches_sixteen_logical_origins(self):
        from training.speculation.dataset import TrainingExample
        from training.speculation.models import DirectRecurrentDrafter
        from training.speculation.objectives import (
            GemmaTargetHead,
            ObjectiveWeights,
            TargetSupport,
        )
        from training.speculation.train import _loss, _slice_example

        origin_count, horizon, target_width, vocabulary = 16, 3, 8, 11
        token_table = torch.randn(vocabulary, target_width, dtype=torch.float16)
        norm = torch.randn(target_width)
        packed_model = DirectRecurrentDrafter(
            target_width, vocabulary, 1, 8, horizon,
            "gated_residual", token_table,
        )
        logical_model = DirectRecurrentDrafter(
            target_width, vocabulary, 1, 8, horizon,
            "gated_residual", token_table,
        )
        logical_model.load_state_dict(packed_model.state_dict())
        packed_head = GemmaTargetHead(token_table, norm, 1e-6, 30.0, chunk_size=5)
        logical_head = GemmaTargetHead(token_table, norm, 1e-6, 30.0, chunk_size=5)
        support_tokens = torch.arange(vocabulary).view(1, 1, -1).expand(
            origin_count, horizon, -1
        ).clone()
        example = TrainingExample(
            context_taps=torch.randn(origin_count, 1, target_width),
            context_positions=torch.arange(origin_count),
            anchor_indices=torch.arange(origin_count),
            rollout_tokens=torch.randint(0, vocabulary, (origin_count, horizon)),
            labels=torch.randint(0, vocabulary, (origin_count, horizon)),
            target_support=TargetSupport(
                token_ids=support_tokens,
                probabilities=torch.full(
                    (origin_count, horizon, vocabulary), 1.0 / vocabulary
                ),
                mask=torch.ones(
                    origin_count, horizon, vocabulary, dtype=torch.bool
                ),
            ),
            target_selected_probabilities=torch.full(
                (origin_count, horizon), 1.0 / vocabulary
            ),
            target_unit_draws=torch.full(
                (origin_count, horizon), 0.5, dtype=torch.float64
            ),
            source_clusters=tuple(f"cluster-{index}" for index in range(origin_count)),
            conversations=tuple(
                f"conversation-{index}" for index in range(origin_count)
            ),
            sequence_ids=tuple(range(origin_count)),
            workloads=tuple(1 + index % 8 for index in range(origin_count)),
            generation_seeds=tuple(17 for _ in range(origin_count)),
            target_positions=torch.arange(origin_count),
            # Canonical trace order: generation seed precedes repetition window.
            sampler=(0.8, 8, 0.95, 0.0, 1.0, 17, 64),
        )
        cell = SimpleNamespace(
            implementation="direct_recurrent",
            parameters=(("curriculum", "deployment_prefix"),
                        ("position_weighting", "dpace")),
        )
        weights = ObjectiveWeights(1.0, 0.1, 0.2, 0.3, 0.0, 0.15)

        packed_loss, _, _ = _loss(
            packed_model, packed_head, example, cell, weights, 0,
            proposal_policy="common_draw",
        )
        (packed_loss * origin_count).backward()
        packed_gradients = {}
        for name, parameter in packed_model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            packed_gradients[name] = parameter.grad.detach().clone() / origin_count

        logical_losses = []
        visited = []
        for index in range(origin_count):
            logical_example = _slice_example(example, index, index + 1)
            visited.extend(logical_example.sequence_ids)
            logical_loss, _, _ = _loss(
                logical_model, logical_head, logical_example, cell, weights, 0,
                proposal_policy="common_draw",
            )
            logical_loss.backward()
            logical_losses.append(logical_loss.detach())
        self.assertEqual(visited, list(example.sequence_ids))
        torch.testing.assert_close(
            packed_loss, torch.stack(logical_losses).mean(), rtol=3e-6, atol=3e-7
        )
        for name, parameter in logical_model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            torch.testing.assert_close(
                packed_gradients[name], parameter.grad / origin_count,
                rtol=3e-5, atol=3e-6,
                msg=lambda message, parameter_name=name: f"{parameter_name}: {message}",
            )

    def test_restart_cosine_uses_remaining_updates_and_binds_its_plan(self):
        from training.speculation.train import (
            _make_optimizer_and_scheduler,
            _scheduler_plan,
            _validate_scheduler_summary,
        )

        contract = self._registry().raw["default_transfer_optimizer"]
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        _, scheduler, plan = _make_optimizer_and_scheduler(
            [parameter], 1, 75, 100, contract
        )
        self.assertEqual(plan["global_start_update"], 75)
        self.assertEqual(plan["span_updates"], 25)
        self.assertEqual(plan["warmup_updates"], 1)
        self.assertEqual(
            plan["base_learning_rate"],
            contract["learning_rate"] * contract["restart_learning_rate_factor"],
        )
        self.assertEqual(scheduler.last_epoch, 0)

        learning_curve = [{
            "optimizer_updates": 75,
            "restart_performed": True,
        }]
        summary = {
            "restarts": 1,
            "update_step": 80,
            "scheduler_plan": plan,
        }
        self.assertEqual(
            _validate_scheduler_summary(summary, learning_curve, 100, contract)[2],
            plan,
        )
        tampered = dict(summary)
        tampered["scheduler_plan"] = _scheduler_plan(1, 74, 100, contract)
        with self.assertRaisesRegex(RuntimeError, "scheduler plan"):
            _validate_scheduler_summary(tampered, learning_curve, 100, contract)

    def test_validation_schedule_uses_fixed_intervals_plus_only_final_update(self):
        from training.speculation.train import _maximum_validation_events

        contract = copy.deepcopy(self._registry().raw["default_transfer_optimizer"])
        # ceil(100 / 16) * 6 = 42 total updates. Four fixed ten-update
        # validations plus the non-aligned final update are admissible; the
        # six epoch boundaries are deliberately not extra patience events.
        contract["validation_interval_steps"] = 10
        self.assertEqual(_maximum_validation_events(100, contract), 5)
        contract["validation_interval_steps"] = 7
        self.assertEqual(_maximum_validation_events(100, contract), 6)

    def test_restarted_optimizer_is_bit_exact_across_checkpoint_resume(self):
        from training.speculation.train import (
            _make_optimizer_and_scheduler,
            _save_checkpoint,
            _validate_scheduler_summary,
        )

        contract = self._registry().raw["default_transfer_optimizer"]
        total_updates = 10
        restart_update = 3
        gradients = [0.1, -0.3, 0.2, 0.5, -0.4, 0.7, -0.1, 0.6, -0.2, 0.8]

        def update(parameter, optimizer, scheduler, gradient):
            optimizer.zero_grad(set_to_none=True)
            parameter.grad = torch.tensor(gradient)
            optimizer.step()
            scheduler.step()

        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer, scheduler, _ = _make_optimizer_and_scheduler(
            [parameter], 0, 0, total_updates, contract
        )
        for gradient in gradients[:restart_update]:
            update(parameter, optimizer, scheduler, gradient)

        optimizer, scheduler, plan = _make_optimizer_and_scheduler(
            [parameter], 1, restart_update, total_updates, contract
        )
        checkpoint_update = 5
        for gradient in gradients[restart_update:checkpoint_update]:
            update(parameter, optimizer, scheduler, gradient)
        learning_curve = [{
            "optimizer_updates": restart_update,
            "restart_performed": True,
        }]
        checkpoint = {
            "model": parameter.detach().clone(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scheduler_plan": plan,
            "restarts": 1,
            "update_step": checkpoint_update,
        }

        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "resume.pt"
            _save_checkpoint(path, checkpoint)
            restored = torch.load(path, map_location="cpu", weights_only=False)

        for gradient in gradients[checkpoint_update:]:
            update(parameter, optimizer, scheduler, gradient)
        uninterrupted_parameter = parameter.detach().clone()
        uninterrupted_optimizer = copy.deepcopy(optimizer.state_dict())
        uninterrupted_scheduler = copy.deepcopy(scheduler.state_dict())

        resumed_parameter = torch.nn.Parameter(restored["model"].clone())
        resumed_optimizer, resumed_scheduler, resumed_plan = _make_optimizer_and_scheduler(
            [resumed_parameter], 1, restart_update, total_updates, contract
        )
        self.assertEqual(resumed_plan, restored["scheduler_plan"])
        _validate_scheduler_summary(restored, learning_curve, total_updates, contract)
        resumed_optimizer.load_state_dict(restored["optimizer"])
        resumed_scheduler.load_state_dict(restored["scheduler"])
        self.assertEqual(
            resumed_scheduler.last_epoch,
            checkpoint_update - restart_update,
        )
        for gradient in gradients[checkpoint_update:]:
            update(resumed_parameter, resumed_optimizer, resumed_scheduler, gradient)

        self.assertTrue(torch.equal(resumed_parameter, uninterrupted_parameter))
        self.assertEqual(resumed_scheduler.state_dict(), uninterrupted_scheduler)
        resumed_state = resumed_optimizer.state_dict()
        self.assertEqual(resumed_state["param_groups"], uninterrupted_optimizer["param_groups"])
        for key, expected in uninterrupted_optimizer["state"][0].items():
            actual = resumed_state["state"][0][key]
            if isinstance(expected, torch.Tensor):
                self.assertTrue(torch.equal(actual, expected), key)
            else:
                self.assertEqual(actual, expected)

    def test_early_stop_completion_preserves_actual_epoch_and_origin_cursor(self):
        from training.speculation.train import _completion_cursor_fields

        # Seven optimizer updates complete one 100-origin epoch at accumulation
        # 16. Stopping after 48 origins of epoch two therefore means update 17,
        # not the fictitious maximum-epoch cursor previously published.
        self.assertEqual(
            _completion_cursor_fields(2, 48, 17, True, 100, 16, 10),
            {
                "terminal_epoch": 2,
                "terminal_origins_in_epoch": 48,
                "stop_requested": True,
            },
        )
        self.assertEqual(
            _completion_cursor_fields(10, 0, 70, False, 100, 16, 10),
            {
                "terminal_epoch": 10,
                "terminal_origins_in_epoch": 0,
                "stop_requested": False,
            },
        )
        with self.assertRaisesRegex(RuntimeError, "partial cursor"):
            _completion_cursor_fields(2, 48, 17, False, 100, 16, 10)
        with self.assertRaisesRegex(RuntimeError, "optimizer updates"):
            _completion_cursor_fields(2, 48, 18, True, 100, 16, 10)

    def test_convergence_accumulates_small_improvements_against_significant_reference(self):
        from training.speculation.train import _advance_convergence

        best = reference = math.inf
        patience = 0
        best, reference, patience, _, significant = _advance_convergence(
            1.0, best, reference, patience, 0.001
        )
        self.assertTrue(significant)
        best, reference, patience, _, significant = _advance_convergence(
            0.9995, best, reference, patience, 0.001
        )
        self.assertFalse(significant)
        self.assertEqual(patience, 1)
        best, reference, patience, _, significant = _advance_convergence(
            0.9989, best, reference, patience, 0.001
        )
        self.assertTrue(significant)
        self.assertEqual(reference, 0.9989)
        self.assertEqual(patience, 0)

        # A 0.125% improvement remains relative even when the loss is below one.
        _, reference, patience, _, significant = _advance_convergence(
            0.19975, 0.2, 0.2, 3, 0.001
        )
        self.assertTrue(significant)
        self.assertEqual(reference, 0.19975)
        self.assertEqual(patience, 0)
        with self.assertRaisesRegex(RuntimeError, "nonfinite"):
            _advance_convergence(math.nan, best, reference, patience, 0.001)

    def test_standard_loss_reuses_objective_logits_without_materializing_metrics(self):
        from training.speculation.train import _loss

        output = object()
        logits = torch.tensor([[[1.0, 2.0]]])
        loss = torch.tensor(0.25)
        example = SimpleNamespace(
            labels=object(), target_support=object(), sampler=object(),
        )
        cell = SimpleNamespace(
            implementation="recurrent",
            parameters=(("position_weighting", "dpace"),),
        )
        with (
            mock.patch("training.speculation.train._rollout", return_value=(output, None)),
            mock.patch(
                "training.speculation.train.standard_objective",
                return_value=(loss, {}, logits),
            ) as objective,
        ):
            actual_loss, metrics, actual_logits = _loss(
                object(), object(), example, cell, object(), 0,
                proposal_policy="deterministic",
                position_weighting_override="uniform",
            )
        self.assertIs(actual_loss, loss)
        self.assertEqual(metrics, {})
        self.assertIs(actual_logits, logits)
        self.assertFalse(objective.call_args.kwargs["materialize_metrics"])
        self.assertEqual(objective.call_args.args[6], "uniform")

    def test_deployment_rollout_is_sequential_and_future_teacher_tokens_are_poisonable(self):
        from training.speculation.models import DraftOutput
        from training.speculation.train import _rollout

        class CausalTokenModel(torch.nn.Module):
            def forward(self, _taps, _positions, _anchors, tokens):
                return DraftOutput(tokens.float().unsqueeze(-1))

        class NextTokenHead(torch.nn.Module):
            def forward(self, states, apply_norm=True):
                del apply_norm
                desired = (states[..., 0].to(torch.int64) + 1) % 8
                vocabulary = torch.arange(8, device=states.device)
                return -(vocabulary - desired.unsqueeze(-1)).abs().float()

        base = SimpleNamespace(
            context_taps=torch.zeros(1, 1, 1),
            context_positions=torch.zeros(1, dtype=torch.int64),
            anchor_indices=torch.zeros(1, dtype=torch.int64),
            rollout_tokens=torch.tensor([[1, 6, 7]], dtype=torch.int64),
            labels=torch.tensor([[2, 3, 4]], dtype=torch.int64),
        )
        output, reached = _rollout(
            CausalTokenModel(), NextTokenHead(), base, "deployment_prefix",
            "deterministic",
        )
        self.assertEqual(output.states[..., 0].tolist(), [[1.0, 2.0, 3.0]])
        self.assertEqual(reached.tolist(), [[True, True, True]])

        poisoned = copy.copy(base)
        poisoned.rollout_tokens = torch.tensor([[1, 0, 5]], dtype=torch.int64)
        poisoned_output, poisoned_reached = _rollout(
            CausalTokenModel(), NextTokenHead(), poisoned, "deployment_prefix",
            "deterministic",
        )
        self.assertTrue(torch.equal(poisoned_output.states, output.states))
        self.assertTrue(torch.equal(poisoned_reached, reached))

        mismatched = copy.copy(base)
        mismatched.labels = torch.tensor([[7, 3, 4]], dtype=torch.int64)
        _, mismatch_reached = _rollout(
            CausalTokenModel(), NextTokenHead(), mismatched, "deployment_prefix",
            "deterministic",
        )
        self.assertEqual(mismatch_reached.tolist(), [[True, False, False]])
        with self.assertRaisesRegex(RuntimeError, "unknown token-prefix curriculum"):
            _rollout(
                CausalTokenModel(), NextTokenHead(), base, "misspelled", "deterministic"
            )

    def test_raw_common_draw_uses_f64_draws_stable_ties_and_strict_cdf_boundaries(self):
        from training.speculation.train import _raw_prefixes

        # Equal logits retain lower token IDs. With two unit weights, exactly
        # 0.5 maps to the second interval because the production sampler tests
        # draw < cumulative rather than draw <= cumulative.
        example = SimpleNamespace(
            labels=torch.tensor([[0, 1]], dtype=torch.int64),
            target_unit_draws=torch.tensor([[0.0, 0.5]], dtype=torch.float64),
            sampler=(1.0, 2, 1.0, 0.0, 1.0, 7, 64),
        )
        logits = torch.tensor([[[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]])
        self.assertEqual(_raw_prefixes(logits, example), [(1, 2)])

        wrong_dtype = copy.copy(example)
        wrong_dtype.target_unit_draws = example.target_unit_draws.float()
        with self.assertRaisesRegex(RuntimeError, "shape/dtype"):
            _raw_prefixes(logits, wrong_dtype)

        unsupported_penalty = copy.copy(example)
        unsupported_penalty.sampler = (1.0, 2, 1.0, 0.0, 1.1, 7, 64)
        with self.assertRaisesRegex(RuntimeError, "history reconstruction"):
            _raw_prefixes(logits, unsupported_penalty)

    def test_root_only_evaluation_B16_matches_B1_per_origin_and_hierarchy(self):
        from training.speculation.manifest import WORKLOADS
        from training.speculation.objectives import ObjectiveWeights
        from training.speculation.train import _evaluate
        from training.speculation.validation_stats import VALIDATION_AGGREGATION

        class Model:
            def eval(self):
                return self

            def train(self):
                return self

        seeds = (11, 22, 33, 44, 55)
        rows = []
        for workload, code in WORKLOADS.items():
            for seed_index, seed in enumerate(seeds):
                for origin in range(15):
                    rows.append(SimpleNamespace(
                        origin_count=1,
                        source_clusters=(f"{code:064x}",),
                        conversations=(f"{100 + code:064x}",),
                        workloads=(code,),
                        generation_seeds=(seed,),
                        labels=torch.zeros(1, 3, dtype=torch.int64),
                        values=torch.tensor([
                            code + seed_index * 0.01 + origin * 0.001
                        ]),
                    ))

        def pack(chunk):
            return SimpleNamespace(
                origin_count=len(chunk),
                source_clusters=tuple(
                    value for row in chunk for value in row.source_clusters
                ),
                conversations=tuple(
                    value for row in chunk for value in row.conversations
                ),
                workloads=tuple(value for row in chunk for value in row.workloads),
                generation_seeds=tuple(
                    value for row in chunk for value in row.generation_seeds
                ),
                labels=torch.cat([row.labels for row in chunk]),
                values=torch.cat([row.values for row in chunk]),
            )

        class Stream:
            context_mode = "root_only"

            def __init__(self, width):
                self.anchor_batch_size = width
                self.batches = [
                    pack(rows[start:start + width])
                    for start in range(0, len(rows), width)
                ]

            def __iter__(self):
                return iter(self.batches)

        def fake_loss(_model, _head, example, _cell, _weights, _epoch, **kwargs):
            self.assertEqual(kwargs["reduction"], "none")
            return (
                example.values.clone(), {},
                torch.zeros(example.origin_count, 3, 1),
            )

        def prefixes(_logits, example):
            return [(1, 2)] * example.origin_count

        def evaluate(width):
            return _evaluate(
                Model(), object(), Stream(width), SimpleNamespace(parameters=()),
                ObjectiveWeights(1.0, 0.1, 0.1, 0.2, 0.0, 0.1),
                torch.device("cpu"), 0,
                workload_weights={workload: 0.125 for workload in WORKLOADS},
                proposal_policy="common_draw",
                checkpoint_metric=(
                    "hierarchical_workload_weighted_exact_accepted_prefix/v1"
                ),
                validation_loss_contract="teacher_prefix_uniform_kl_ce_all_edges",
                validation_aggregation=VALIDATION_AGGREGATION,
                required_generation_seeds=seeds,
            )

        with (
            mock.patch("training.speculation.train._to_device", side_effect=lambda item, _: item),
            mock.patch("training.speculation.train._loss", side_effect=fake_loss),
            mock.patch("training.speculation.train._raw_prefixes", side_effect=prefixes),
        ):
            result = evaluate(16)
            logical = evaluate(1)
        self.assertEqual(result, logical)
        self.assertEqual(result["window_count"], 600)
        self.assertEqual(
            result["workload_window_count"],
            {workload: 75 for workload in WORKLOADS},
        )
        self.assertEqual(set(result["workload_mean_loss"]), set(WORKLOADS))
        for workload, code in WORKLOADS.items():
            self.assertAlmostEqual(
                result["workload_mean_loss"][workload],
                float(code) + 0.027,
                places=6,
            )
        self.assertAlmostEqual(result["origin_mean_loss"], 4.527, places=6)
        self.assertAlmostEqual(
            result["workload_weighted_mean_loss"], 4.527, places=6
        )
        self.assertAlmostEqual(result["mean_loss"], 4.527, places=6)
        self.assertEqual(
            result["validation_loss_contract"],
            "teacher_prefix_uniform_kl_ce_all_edges",
        )
        self.assertEqual(
            result["checkpoint_selection_metric"],
            "hierarchical_workload_weighted_exact_accepted_prefix/v1",
        )
        self.assertEqual(result["proposal_policy"], "common_draw")
        self.assertEqual(result["checkpoint_selection_value"], 2.0)
        self.assertEqual(result["deterministic_workload_weighted_mean_prefix"], 1.0)
        self.assertEqual(result["common_draw_workload_weighted_mean_prefix"], 2.0)
        self.assertEqual(result["hierarchical_validation"]["conversation_seed_count"], 40)
        self.assertEqual(result["hierarchical_validation"]["conversation_count"], 8)

    def test_evaluation_fails_closed_on_missing_workload_or_nonfinite_loss(self):
        from training.speculation.manifest import WORKLOADS
        from training.speculation.objectives import ObjectiveWeights
        from training.speculation.train import _evaluate
        from training.speculation.validation_stats import VALIDATION_AGGREGATION

        class Model:
            def eval(self):
                return self

            def train(self):
                return self

        seeds = (11, 22, 33, 44, 55)
        weights = {workload: 0.125 for workload in WORKLOADS}

        class RootStream:
            context_mode = "root_only"
            anchor_batch_size = 16

            def __iter__(self):
                return iter([SimpleNamespace(
                    origin_count=2,
                    labels=torch.zeros(2, 3, dtype=torch.int64),
                )])

        def malformed_loss(shape):
            def implementation(*_args, **kwargs):
                self.assertEqual(kwargs["reduction"], "none")
                return torch.zeros(shape), {}, torch.zeros(2, 3, 1)
            return implementation

        for shape in ((), (1,), (2, 1)):
            with self.subTest(loss_shape=shape), mock.patch(
                "training.speculation.train._to_device", side_effect=lambda item, _: item
            ), mock.patch(
                "training.speculation.train._loss", side_effect=malformed_loss(shape)
            ), self.assertRaisesRegex(RuntimeError, "loss/batch shape"):
                _evaluate(
                    Model(), object(), RootStream(), SimpleNamespace(parameters=()),
                    ObjectiveWeights(1.0, 0.1, 0.1, 0.2, 0.0, 0.1),
                    torch.device("cpu"), 0, workload_weights=weights,
                    proposal_policy="common_draw",
                    checkpoint_metric=(
                        "hierarchical_workload_weighted_exact_accepted_prefix/v1"
                    ),
                    validation_loss_contract=(
                        "teacher_prefix_uniform_kl_ce_all_edges"
                    ),
                    validation_aggregation=VALIDATION_AGGREGATION,
                    required_generation_seeds=seeds,
                )

        base = SimpleNamespace(
            origin_count=1, source_clusters=(f"{1:064x}",),
            conversations=(f"{101:064x}",), workloads=(1,),
            generation_seeds=(11,), labels=torch.zeros(1, 3, dtype=torch.int64),
            value=math.nan,
        )

        def fake_loss(_model, _head, example, _cell, _weights, _epoch, **_kwargs):
            return torch.tensor(example.value), {}, torch.zeros(1, 3, 1)

        patches = (
            mock.patch("training.speculation.train._to_device", side_effect=lambda item, _: item),
            mock.patch("training.speculation.train._loss", side_effect=fake_loss),
            mock.patch("training.speculation.train._raw_prefixes", return_value=[(0, 0)]),
        )
        with patches[0], patches[1], patches[2], self.assertRaisesRegex(
            RuntimeError, "nonfinite"
        ):
            _evaluate(
                Model(), object(), [base], SimpleNamespace(parameters=()),
                ObjectiveWeights(1.0, 0.1, 0.1, 0.2, 0.0, 0.1),
                torch.device("cpu"), 0,
                workload_weights=weights,
                proposal_policy="common_draw",
                checkpoint_metric=(
                    "hierarchical_workload_weighted_exact_accepted_prefix/v1"
                ),
                validation_loss_contract="teacher_prefix_uniform_kl_ce_all_edges",
                validation_aggregation=VALIDATION_AGGREGATION,
                required_generation_seeds=seeds,
            )

        missing = SimpleNamespace(
            origin_count=1, source_clusters=(f"{1:064x}",), value=0.5,
            labels=torch.zeros(1, 3, dtype=torch.int64),
        )
        patches = (
            mock.patch("training.speculation.train._to_device", side_effect=lambda item, _: item),
            mock.patch("training.speculation.train._loss", side_effect=fake_loss),
            mock.patch("training.speculation.train._raw_prefixes", return_value=[(0, 0)]),
        )
        with patches[0], patches[1], patches[2], self.assertRaisesRegex(
            RuntimeError, "hierarchical identity"
        ):
            _evaluate(
                Model(), object(), [missing], SimpleNamespace(parameters=()),
                ObjectiveWeights(1.0, 0.1, 0.1, 0.2, 0.0, 0.1),
                torch.device("cpu"), 0,
                workload_weights=weights,
                proposal_policy="common_draw",
                checkpoint_metric=(
                    "hierarchical_workload_weighted_exact_accepted_prefix/v1"
                ),
                validation_loss_contract="teacher_prefix_uniform_kl_ce_all_edges",
                validation_aggregation=VALIDATION_AGGREGATION,
                required_generation_seeds=seeds,
            )

        covered = [SimpleNamespace(
            origin_count=1, source_clusters=(f"{1:064x}",),
            conversations=(f"{101:064x}",), workloads=(1,),
            generation_seeds=(seed,), labels=torch.zeros(1, 3, dtype=torch.int64),
            value=0.5,
        ) for seed in seeds]
        patches = (
            mock.patch("training.speculation.train._to_device", side_effect=lambda item, _: item),
            mock.patch("training.speculation.train._loss", side_effect=fake_loss),
            mock.patch("training.speculation.train._raw_prefixes", return_value=[(0, 0)]),
        )
        with patches[0], patches[1], patches[2], self.assertRaisesRegex(
            RuntimeError, "every frozen workload"
        ):
            _evaluate(
                Model(), object(), covered, SimpleNamespace(parameters=()),
                ObjectiveWeights(1.0, 0.1, 0.1, 0.2, 0.0, 0.1),
                torch.device("cpu"), 0,
                workload_weights=weights,
                proposal_policy="common_draw",
                checkpoint_metric=(
                    "hierarchical_workload_weighted_exact_accepted_prefix/v1"
                ),
                validation_loss_contract="teacher_prefix_uniform_kl_ce_all_edges",
                validation_aggregation=VALIDATION_AGGREGATION,
                required_generation_seeds=seeds,
            )

    def test_best_checkpoint_requires_bound_digest_identity_and_selection_value(self):
        from training.speculation.corpus import file_sha256
        from training.speculation.train import (
            _load_best_checkpoint,
            _publish_best_checkpoint,
        )

        with tempfile.TemporaryDirectory() as temporary:
            filename = "cell.best.000000000000.pt"
            path = pathlib.Path(temporary) / filename
            state = {
                "schema": "local-gguf-engine/training-best/v2",
                "checkpoint_selection_value": 0.25,
                "identity": "expected",
                "best_checkpoint_file": filename,
                "model": {"weight": torch.tensor([1.0])},
            }
            _publish_best_checkpoint(path, state)
            digest = file_sha256(path)

            def validate_identity(value, _description):
                if value.get("identity") != "expected":
                    raise RuntimeError("stale")

            loaded = _load_best_checkpoint(
                path, torch.device("cpu"), digest, 0.25, filename, validate_identity
            )
            self.assertEqual(loaded["checkpoint_selection_value"], 0.25)
            with self.assertRaisesRegex(RuntimeError, "digest mismatch"):
                _load_best_checkpoint(
                    path, torch.device("cpu"), "0" * 64, 0.25,
                    filename, validate_identity,
                )
            with self.assertRaisesRegex(RuntimeError, "identity/selection value"):
                _load_best_checkpoint(
                    path, torch.device("cpu"), digest, 0.5,
                    filename, validate_identity,
                )

    def test_versioned_best_publication_survives_interruption_and_exact_replay(self):
        from training.speculation.corpus import file_sha256
        from training.speculation.train import (
            _best_checkpoint_filename,
            _load_best_checkpoint,
            _publish_best_checkpoint,
            _save_checkpoint,
            _validate_best_checkpoint_files,
        )

        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            cell_id = "cell"
            maximum = 4
            old_filename = _best_checkpoint_filename(cell_id, 0, maximum)
            old_path = output / old_filename
            old_checkpoint = {
                "schema": "local-gguf-engine/training-best/v2",
                "checkpoint_selection_value": 0.5,
                "identity": "expected",
                "best_checkpoint_file": old_filename,
                "model": {"weight": torch.tensor([1.0])},
            }
            old_digest = _publish_best_checkpoint(old_path, old_checkpoint)
            state_path = output / "cell.state.pt"
            _save_checkpoint(state_path, {
                "best_checkpoint_file": old_filename,
                "best_checkpoint_sha256": old_digest,
                "best": 0.5,
                "learning_curve": [{"mean_loss": 0.5}],
            })

            # Simulate interruption after a new immutable best file is durable,
            # but before the state rename commits its reference.
            new_filename = _best_checkpoint_filename(cell_id, 1, maximum)
            new_path = output / new_filename
            new_checkpoint = {
                "schema": "local-gguf-engine/training-best/v2",
                "checkpoint_selection_value": 0.4,
                "identity": "expected",
                "best_checkpoint_file": new_filename,
                "model": {"weight": torch.tensor([2.0])},
            }
            new_digest = _publish_best_checkpoint(new_path, new_checkpoint)

            # The smaller link-cleanup crash window is recoverable too: both
            # names identify the same already-fsynced immutable bytes.
            cleanup_partial = new_path.with_suffix(new_path.suffix + ".partial")
            cleanup_partial.hardlink_to(new_path)

            committed = torch.load(state_path, map_location="cpu", weights_only=False)
            self.assertEqual(committed["best_checkpoint_file"], old_filename)
            self.assertEqual(committed["best_checkpoint_sha256"], old_digest)
            _validate_best_checkpoint_files(
                output, cell_id, maximum, len(committed["learning_curve"]),
                committed["best_checkpoint_file"],
            )
            self.assertFalse(cleanup_partial.exists())

            def validate_identity(value, _description):
                if value.get("identity") != "expected":
                    raise RuntimeError("stale")

            loaded = _load_best_checkpoint(
                old_path, torch.device("cpu"), old_digest, 0.5,
                old_filename, validate_identity,
            )
            self.assertEqual(float(loaded["model"]["weight"][0]), 1.0)

            # Deterministic replay publishes the byte-identical candidate and
            # safely reuses the orphan instead of overwriting it.
            self.assertEqual(
                _publish_best_checkpoint(new_path, new_checkpoint), new_digest
            )
            self.assertEqual(old_digest, file_sha256(old_path))

            conflicting = copy.deepcopy(new_checkpoint)
            conflicting["model"]["weight"] = torch.tensor([3.0])
            with self.assertRaisesRegex(RuntimeError, "conflicts with published content"):
                _publish_best_checkpoint(new_path, conflicting)
            self.assertEqual(new_digest, file_sha256(new_path))

    def test_best_checkpoint_filename_and_file_set_fail_closed(self):
        from training.speculation.train import (
            _best_checkpoint_filename,
            _resolve_best_checkpoint_path,
            _validate_best_checkpoint_files,
        )

        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "validation index"):
                _best_checkpoint_filename("cell", 2, 2)
            for malformed in (
                "../cell.best.000000000000.pt",
                "other.best.000000000000.pt",
                "cell.best.0.pt",
            ):
                with self.subTest(malformed=malformed), self.assertRaisesRegex(
                    RuntimeError, "filename"
                ):
                    _resolve_best_checkpoint_path(output, "cell", malformed, 2)

            future = output / "cell.best.000000000001.pt"
            future.touch()
            with self.assertRaisesRegex(RuntimeError, "future best checkpoint"):
                _validate_best_checkpoint_files(output, "cell", 2, 0, None)

    def test_checkpoint_identity_binds_every_run_contract_decision(self):
        from training.speculation.corpus import file_sha256
        from training.speculation.run_contract import load_run_contract
        from training.speculation.train import train_cell, training_code_digest

        class References:
            def __init__(self, count):
                self.count = count

            def __len__(self):
                return self.count

        registry = self._registry()
        run_contract = load_run_contract(
            ROOT / "training/speculation/phase5_h3_control_run.json",
            registry,
        )
        cell_contract = run_contract.cells[0]
        cell = next(
            candidate for candidate in registry.cells()
            if candidate.identifier == cell_contract.cell_id
        )
        scale = 100000
        source_digest = "aa" * 32
        trace_digest = "bb" * 32
        training_membership = SimpleNamespace(
            trace_corpus_sha256=trace_digest,
            horizon=3,
            partition="train",
            tap_set=("final",),
            references=References(scale),
            sha256="cc" * 32,
        )
        validation_membership = SimpleNamespace(
            trace_corpus_sha256=trace_digest,
            horizon=3,
            partition="validation",
            tap_set=("final",),
            references=References(1),
            sha256="dd" * 32,
        )
        catalog = SimpleNamespace(
            registry_sha256=registry.digest,
            trace_corpus_sha256=trace_digest,
        )

        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary)
            model_path = output / "model.gguf"
            model_path.write_bytes(b"model identity fixture")
            receipt_path = output / f"{cell.identifier}.training.json"
            identity = {
                "registry_sha256": registry.digest,
                "cell_id": cell.identifier,
                "model_sha256": file_sha256(model_path),
                "training_code_sha256": training_code_digest(ROOT),
                "scale": scale,
                "source_manifest_sha256": source_digest,
                "trace_corpus_sha256": trace_digest,
                "training_membership_sha256": training_membership.sha256,
                "training_origin_count": scale,
                "validation_membership_sha256": validation_membership.sha256,
                "validation_origin_count": 1,
                "run_contract_sha256": run_contract.digest,
                "proposal_policy": cell_contract.proposal_policy,
                "dpace_alpha": cell_contract.dpace_alpha,
                "data_order_seed": run_contract.data_order_seed,
                "data_order_derivation": run_contract.data_order_derivation,
                "target_architecture": run_contract.target_architecture,
                "target_embedding_width": run_contract.target_embedding_width,
                "target_vocabulary": run_contract.target_vocabulary,
                "target_output_head": run_contract.target_output_head,
                "checkpoint_selection_metric": run_contract.checkpoint_metric,
                "validation_loss_contract": run_contract.validation_loss,
                "validation_aggregation": run_contract.validation_aggregation,
                "validation_cadence": run_contract.validation_cadence,
                "minimum_validation_source_clusters_per_workload": (
                    run_contract.minimum_validation_source_clusters_per_workload
                ),
                "root_only_interleave_algorithm": (
                    run_contract.root_only_interleave_algorithm
                ),
                "root_only_interleave_clusters_per_sampler": (
                    run_contract.root_only_interleave_clusters_per_sampler
                ),
                "root_only_origin_shuffle_buffer": (
                    run_contract.root_only_origin_shuffle_buffer
                ),
                "physical_validation_batch_size": 16,
            }

            def invoke(receipt):
                receipt_path.write_text(json.dumps(receipt))
                return train_cell(
                    ROOT, registry, cell, output / "traces", output, model_path,
                    identity["model_sha256"], scale, torch.device("cpu"),
                    source_digest, trace_digest,
                    catalog, training_membership, validation_membership,
                    run_contract, cell_contract,
                )

            # The shared identity validator accepts every exact contract field;
            # this deliberately incomplete receipt then fails at the next,
            # unrelated integrity boundary.
            with self.assertRaisesRegex(RuntimeError, "receipt/checkpoint integrity"):
                invoke({"schema": "deliberately-incomplete", **identity})

            replacements = {
                "run_contract_sha256": "0" * 64,
                "proposal_policy": "deterministic",
                "dpace_alpha": 0.1,
                "data_order_seed": run_contract.data_order_seed + 1,
                "data_order_derivation": "seed_plus_epoch",
                "target_architecture": "llama",
                "target_embedding_width": 2048,
                "target_vocabulary": 32000,
                "target_output_head": "output_weight_f16",
                "checkpoint_selection_metric": "diagnostic_loss",
                "validation_loss_contract": "deployment_masked",
                "validation_aggregation": "origin_mean/v0",
                "validation_cadence": "interval_plus_epoch_end/v0",
                "minimum_validation_source_clusters_per_workload": 9,
                "root_only_interleave_algorithm": "drain-one-response/v0",
                "root_only_interleave_clusters_per_sampler": 8,
                "root_only_origin_shuffle_buffer": 2048,
                "physical_validation_batch_size": 8,
            }
            for field, replacement in replacements.items():
                with self.subTest(field=field), self.assertRaisesRegex(
                    RuntimeError, "completed receipt identity is stale"
                ):
                    invoke({
                        "schema": "deliberately-incomplete",
                        **identity,
                        field: replacement,
                    })

            missing = dict(identity)
            del missing["run_contract_sha256"]
            with self.assertRaisesRegex(RuntimeError, "identity is incomplete"):
                invoke({"schema": "deliberately-incomplete", **missing})

    def test_paper_supported_dpace_alpha_overrides_local_registry_default(self):
        from training.speculation.train import _objective_weights

        registry = self._registry()
        self.assertEqual(registry.raw["transfer_objective"]["dpace_alpha"], 0.1)
        self.assertEqual(_objective_weights(registry, 0.5).dpace_alpha, 0.5)
        for malformed in (0.1, math.nan, True):
            with self.subTest(malformed=malformed), self.assertRaisesRegex(
                RuntimeError, "paper-supported D-PACE"
            ):
                _objective_weights(registry, malformed)

    def test_three_model_seeds_share_one_frozen_train_and_validation_membership(self):
        from training.speculation.train import train_cells

        registry = self._registry()
        wanted_parameters = {
            "cell": "gated_residual", "context_mode": "root_only",
            "curriculum": "deployment_prefix", "horizon": 3,
            "position_weighting": "dpace", "selected_head_precision": "f16",
            "selector": "full_vocabulary", "tap_set": ["final"], "width": 512,
        }
        cells = [
            cell for cell in registry.cells()
            if cell.unit_id == "R-direct-recurrent-transfer"
            and dict(cell.parameters) == wanted_parameters
        ]
        self.assertEqual(len(cells), 3)
        run_contract_path = ROOT / "training/speculation/phase5_h3_control_run.json"
        catalog = SimpleNamespace(trace_corpus_sha256="cc" * 32)
        train_membership = SimpleNamespace(name="train")
        validation_membership = SimpleNamespace(name="validation")

        def membership(
            _catalog, horizon, partition, maximum, _weights, _seeds, tap_set
        ):
            self.assertEqual(horizon, 3)
            self.assertEqual(tap_set, ("final",))
            if partition == "train":
                self.assertEqual(maximum, 100000)
                return train_membership
            self.assertIsNone(maximum)
            return validation_membership

        with (
            mock.patch(
                "training.speculation.train.validate_trace_corpus_with_index",
                return_value=({
                    "source_manifest_sha256": "aa" * 32,
                    "trace_corpus_sha256": "cc" * 32,
                    "model_sha256": "dd" * 32,
                }, catalog),
            ),
            mock.patch(
                "training.speculation.train.build_origin_membership",
                side_effect=membership,
            ) as build,
            mock.patch(
                "training.speculation.train.validate_minimum_workload_clusters"
            ) as minimum_clusters,
            mock.patch("training.speculation.train._device", return_value=torch.device("cpu")),
            mock.patch("training.speculation.train.train_cell", return_value={}) as train,
        ):
            results = train_cells(
                ROOT, registry, pathlib.Path("traces"), pathlib.Path("output"),
                pathlib.Path("model"), 100000, set(), set(),
                {cell.identifier for cell in cells}, "cpu", True,
                pathlib.Path("receipt"), None, run_contract_path,
            )
        self.assertEqual(len(results), 3)
        self.assertEqual(build.call_count, 2)
        minimum_clusters.assert_called_once_with(validation_membership, 10)
        self.assertEqual(train.call_count, 3)
        shared_contracts = []
        ordered_cell_contracts = []
        for call in train.call_args_list:
            self.assertEqual(call.args[6], "dd" * 32)
            self.assertIs(call.args[-5], catalog)
            self.assertIs(call.args[-4], train_membership)
            self.assertIs(call.args[-3], validation_membership)
            shared_contracts.append(call.args[-2])
            ordered_cell_contracts.append(call.args[-1])
        self.assertTrue(all(contract is shared_contracts[0] for contract in shared_contracts))
        self.assertEqual(shared_contracts[0].data_order_seed, 10723151780598845931)
        self.assertNotIn(
            shared_contracts[0].data_order_seed,
            {cell.seed for cell in cells},
        )
        self.assertEqual(
            [contract.cell_id for contract in ordered_cell_contracts],
            [contract.cell_id for contract in shared_contracts[0].cells],
        )
        self.assertEqual(
            {contract.proposal_policy for contract in ordered_cell_contracts},
            {"common_draw"},
        )
        self.assertEqual(
            {contract.dpace_alpha for contract in ordered_cell_contracts},
            {0.5},
        )

    def test_training_selection_must_exactly_equal_run_contract_cells(self):
        from training.speculation.run_contract import load_run_contract
        from training.speculation.train import train_cells

        registry = self._registry()
        run_contract_path = ROOT / "training/speculation/phase5_h3_control_run.json"
        contract = load_run_contract(run_contract_path, registry)
        contracted = {cell.cell_id for cell in contract.cells}
        outside = next(
            cell.identifier for cell in registry.cells()
            if cell.identifier not in contracted
        )
        selections = (
            set(),
            {contract.cells[0].cell_id},
            contracted | {outside},
        )
        for selection in selections:
            with self.subTest(selection_count=len(selection)), mock.patch(
                "training.speculation.train.validate_trace_corpus_with_index"
            ) as validate_corpus, self.assertRaisesRegex(
                RuntimeError, "exactly equal the run-contract cells"
            ):
                train_cells(
                    ROOT, registry, pathlib.Path("traces"), pathlib.Path("output"),
                    pathlib.Path("model"), 100000, set(), set(), selection,
                    "cpu", True, pathlib.Path("receipt"), None,
                    run_contract_path,
                )
            validate_corpus.assert_not_called()

    def test_rng_checkpoint_restores_python_numpy_and_torch(self):
        from training.speculation.train import _restore_rng_state, _rng_state

        random.seed(11)
        numpy.random.seed(12)
        torch.manual_seed(13)
        state = _rng_state(torch.device("cpu"))
        expected = (random.random(), float(numpy.random.random()), float(torch.rand(())))
        for _ in range(5):
            random.random()
            numpy.random.random()
            torch.rand(())
        _restore_rng_state(state, torch.device("cpu"))
        actual = (random.random(), float(numpy.random.random()), float(torch.rand(())))
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
