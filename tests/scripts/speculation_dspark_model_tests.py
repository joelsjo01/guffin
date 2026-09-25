import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "offline PyTorch training dependencies are not installed")
class DSparkModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)

    def config(self, **overrides):
        from training.speculation.dspark import DSparkConfig
        values = dict(target_width=8, vocabulary=19, tap_count=3, width=8,
                      layers=2, horizon=3, rank=4, correction="markov",
                      attention_heads=4, kv_heads=2, head_dim=4,
                      intermediate_width=20, mask_token_id=0, rms_epsilon=1e-6,
                      rope_theta=10000.0, embedding_scale=1.0, initializer_std=0.02)
        values.update(overrides)
        return DSparkConfig(**values)

    def model(self, **overrides):
        from training.speculation.dspark import DSparkDrafter
        cfg = self.config(**overrides)
        return DSparkDrafter(cfg, torch.randn(cfg.vocabulary, cfg.target_width))

    def inputs(self):
        return (torch.randn(6, 3, 8), torch.arange(11, 17),
                torch.tensor([1, 3]), torch.tensor([[2, 4, 5], [3, 4, 4]]))

    def test_full_backbone_matches_independent_source_equations(self):
        # Source equation golden, deliberately not calling model norm, RoPE,
        # attention, or layer helpers in the independent side.
        model = self.model()
        taps, positions, anchors, rollout = self.inputs()
        cfg = model.config
        actual = model.prepare_backbone(taps, positions, anchors, rollout[:, 0])

        def norm(x, layer):
            work = x.float()
            return layer.weight * (work / torch.sqrt(
                work.square().mean(-1, keepdim=True) + cfg.rms_epsilon)).to(x.dtype)

        def linear(x, layer):
            return x @ layer.weight.T

        def rotate(x, pos):
            result = torch.empty_like(x)
            half = cfg.head_dim // 2
            for axis in range(half):
                angle = pos.float() / cfg.rope_theta ** (2.0 * axis / cfg.head_dim)
                c, s = angle.cos()[:, None, :], angle.sin()[:, None, :]
                result[..., axis] = x[..., axis] * c - x[..., axis + half] * s
                result[..., axis + half] = x[..., axis + half] * c + x[..., axis] * s
            return result

        context = norm(linear(taps.flatten(-2), model.fusion), model.fusion_norm)
        ids = torch.zeros(2, 3, dtype=torch.int64)
        ids[:, 0] = rollout[:, 0]
        block = model.token_table[ids]
        block_positions = positions[anchors, None] + 1 + torch.arange(cfg.horizon)
        for layer in model.layers:
            normalized = norm(block, layer.input_norm)
            q = norm(linear(normalized, layer.query).reshape(
                2, 3, cfg.attention_heads, cfg.head_dim), layer.query_norm).transpose(1, 2)
            ctx = context.unsqueeze(0).expand(2, -1, -1)
            joined = torch.cat((ctx, normalized), dim=1)
            k = norm(linear(joined, layer.key).reshape(
                2, 9, cfg.kv_heads, cfg.head_dim), layer.key_norm).transpose(1, 2)
            v = linear(joined, layer.value).reshape(
                2, 9, cfg.kv_heads, cfg.head_dim).transpose(1, 2)
            key_positions = torch.cat((positions[None].expand(2, -1), block_positions), dim=1)
            q, k = rotate(q, block_positions), rotate(k, key_positions)
            heads = []
            for batch in range(2):
                head_outputs = []
                visible = torch.cat((positions < block_positions[batch, 0],
                                     torch.ones(3, dtype=torch.bool)))
                for head in range(cfg.attention_heads):
                    group = head // (cfg.attention_heads // cfg.kv_heads)
                    score = q[batch, head] @ k[batch, group].T / cfg.head_dim ** 0.5
                    score[:, ~visible] = -torch.inf
                    head_outputs.append(torch.softmax(score, -1) @ v[batch, group])
                heads.append(torch.stack(head_outputs, 1).flatten(-2))
            block = block + linear(torch.stack(heads), layer.output)
            normalized = norm(block, layer.ffn_norm)
            gate = linear(normalized, layer.gate)
            block = block + linear(gate * torch.sigmoid(gate) * linear(normalized, layer.up), layer.down)
        expected = norm(block, model.final_norm)
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)

    def test_positions_masks_and_future_features_are_causal(self):
        model = self.model()
        taps, positions, anchors, rollout = self.inputs()
        captured = []
        handle = model.layers[0].register_forward_pre_hook(lambda _, args: captured.append(args))
        actual = model.prepare_backbone(taps, positions, anchors, rollout[:, 0])
        handle.remove()
        input_block, _, block_positions, _, visible = captured[0]
        torch.testing.assert_close(block_positions, torch.tensor([[13, 14, 15], [15, 16, 17]]))
        torch.testing.assert_close(input_block[:, 1], model.token_table[0].expand(2, -1))
        torch.testing.assert_close(input_block[:, 1], input_block[:, 2], rtol=0, atol=0)
        self.assertEqual(visible.tolist(), [[True, True, False, False, False, False],
                                          [True, True, True, True, False, False]])
        changed = taps.clone()
        changed[4:] += 100
        future = model.prepare_backbone(changed, positions, anchors, rollout[:, 0])
        torch.testing.assert_close(actual, future, rtol=0, atol=0)
        changed = taps.clone()
        changed[0] *= -3
        past = model.prepare_backbone(changed, positions, anchors, rollout[:, 0])
        self.assertFalse(torch.equal(actual, past))
        taps.requires_grad_()
        model.prepare_backbone(taps, positions, anchors, rollout[:, 0]).square().sum().backward()
        self.assertEqual(torch.count_nonzero(taps.grad[4:]).item(), 0)

    def test_markov_gated_and_rnn_vocabulary_equations(self):
        for correction in ("markov", "gated", "source_rnn"):
            with self.subTest(correction=correction):
                model = self.model(correction=correction)
                base = torch.randn(2, 3, 8)
                tokens = torch.tensor([[2, 5, 2], [3, 3, 7]])
                out = model.correct(base, tokens)
                embeddings = model.token_rank.weight[tokens]
                if correction == "markov":
                    expected = embeddings
                elif correction == "gated":
                    joined = torch.cat((base, embeddings), -1)
                    expected = embeddings * torch.sigmoid(
                        joined @ model.correction_gate.weight.T + model.correction_gate.bias)
                else:
                    state = torch.zeros(2, 4)
                    outputs = []
                    for depth in range(3):
                        z = torch.cat((state, embeddings[:, depth], base[:, depth]), -1)
                        raw = z @ model.correction_joint.weight.T + model.correction_joint.bias
                        g, c, o = raw.chunk(3, dim=-1)
                        state = g.sigmoid() * state + (1 - g.sigmoid()) * c.tanh()
                        outputs.append(o.tanh())
                    expected = torch.stack(outputs, 1)
                torch.testing.assert_close(out.correction_features, expected, rtol=0, atol=0)
                torch.testing.assert_close(out.correction_features @ out.correction_weight.T,
                                           expected @ model.correction_output.weight.T)
                self.assertEqual(out.correction_weight.shape, (19, 4))
                self.assertIsNone(model.correction_output.bias)
                torch.testing.assert_close(out.base_states, base, rtol=0, atol=0)
                confidence = torch.cat((base, embeddings), -1) @ model.confidence.weight.T
                confidence = (confidence + model.confidence.bias).squeeze(-1)
                torch.testing.assert_close(out.confidence, confidence)

    def test_full_gemma_transfer_shape_is_not_silently_reduced(self):
        from training.speculation.dspark import DSparkDrafter
        config = self.config(target_width=3584, vocabulary=256000, tap_count=5,
                             width=3584, layers=5, horizon=7, rank=256,
                             attention_heads=16, kv_heads=8, head_dim=256,
                             intermediate_width=14336, embedding_scale=3584 ** 0.5)
        with torch.device("meta"):
            model = DSparkDrafter(config, torch.empty(256000, 3584))
        self.assertEqual(len(model.layers), 5)
        self.assertEqual(model.fusion.weight.shape, (3584, 5 * 3584))
        self.assertEqual(model.layers[0].query.weight.shape, (4096, 3584))
        self.assertEqual(model.layers[0].key.weight.shape, (2048, 3584))
        self.assertEqual(model.layers[0].up.weight.shape, (14336, 3584))
        self.assertEqual(model.correction_output.weight.shape, (256000, 256))
        self.assertIsInstance(model.token_projection, torch.nn.Identity)
        self.assertIsInstance(model.to_target, torch.nn.Identity)
        self.assertNotIn("token_table", model.state_dict())

    def test_gru_is_separate_local_equation_and_rnn_remembers_earlier_tokens(self):
        model = self.model(correction="gru")
        base = torch.randn(2, 3, 8)
        tokens = torch.tensor([[2, 5, 2], [3, 3, 7]])
        features = torch.cat((base, model.token_rank.weight[tokens]), -1)
        gru = model.correction_gru
        state = torch.zeros(2, 4)
        expected = []
        for depth in range(3):
            ir, iz, inn = (features[:, depth] @ gru.weight_ih.T + gru.bias_ih).chunk(3, -1)
            hr, hz, hn = (state @ gru.weight_hh.T + gru.bias_hh).chunk(3, -1)
            reset, update = (ir + hr).sigmoid(), (iz + hz).sigmoid()
            candidate = (inn + reset * hn).tanh()
            state = (1 - update) * candidate + update * state
            expected.append(state)
        torch.testing.assert_close(model.correct(base, tokens).correction_features,
                                   torch.stack(expected, 1))
        rnn = self.model(correction="source_rnn")
        original = rnn.correct(base, tokens).correction_features
        changed = tokens.clone()
        changed[:, 0] += 1
        later = rnn.correct(base, changed).correction_features
        self.assertFalse(torch.equal(original[:, 2], later[:, 2]))
        markov = self.model()
        torch.testing.assert_close(markov.correct(base, tokens).correction_features[:, 2],
                                   markov.correct(base, changed).correction_features[:, 2],
                                   rtol=0, atol=0)

    def test_all_horizons_zero_correction_and_explicit_compact_transfer(self):
        for horizon in (3, 4, 7, 8, 15, 16, 23):
            model = self.model(horizon=horizon, correction="none", rank=0,
                               layers=1, width=12)
            taps, pos, anchors, _ = self.inputs()
            out = model(taps, pos, anchors, torch.ones(2, horizon, dtype=torch.int64))
            self.assertEqual(out.base_states.shape, (2, horizon, 8))
            self.assertEqual(out.confidence.shape, (2, horizon))
            self.assertIsNone(out.correction_features)
            self.assertIsNone(out.correction_weight)
            self.assertTrue(out.head_pre_normalized)

    def test_invalid_shapes_positions_tokens_and_nonfinite_fail_closed(self):
        for changes in ({"kv_heads": 3}, {"head_dim": 3}, {"layers": 0},
                        {"horizon": 24}, {"layers": 9}, {"tap_count": 45}, {"rank": 513},
                        {"mask_token_id": 19}, {"rank": 0}, {"correction": "rnn"},
                        {"embedding_scale": float("nan")}, {"rope_theta": 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.config(**changes)
        model = self.model()
        taps, pos, anchors, rollout = self.inputs()
        for bad_positions in (pos.float(), pos * 2, pos - 20, pos + 2**24):
            with self.assertRaises(ValueError):
                model(taps, bad_positions, anchors, rollout)
        for bad_tokens in (rollout.float(), rollout + 19, rollout - 20):
            with self.assertRaises(ValueError):
                model(taps, pos, anchors, bad_tokens)
        with self.assertRaises(ValueError):
            model(taps, pos, torch.tensor([8, 3]), rollout)
        taps[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            model(taps, pos, anchors, rollout)
        with self.assertRaises(ValueError):
            model.correct(torch.full((2, 3, 8), float("inf")), rollout)
        with self.assertRaises(ValueError):
            model.correct(torch.empty((0, 3, 8)), rollout[:0])

    def test_initialization_and_one_parallel_pass_with_differentiable_correction(self):
        model = self.model(correction="source_rnn")
        self.assertGreater(torch.count_nonzero(model.correction_output.weight).item(), 0)
        torch.testing.assert_close(model.correction_joint.bias,
                                   torch.zeros_like(model.correction_joint.bias), rtol=0, atol=0)
        before = model.token_table.clone()
        taps, pos, anchors, tokens = self.inputs()
        calls = []
        handle = model.layers[0].register_forward_hook(lambda *_: calls.append(1))
        base = model.prepare_backbone(taps, pos, anchors, tokens[:, 0])
        for _ in range(3):
            model.correct(base, tokens)
        out = model.correct(base, tokens)
        loss = out.base_states.square().sum() + out.confidence.sum()
        loss = loss + (out.correction_features @ out.correction_weight.T).square().sum()
        loss.backward()
        handle.remove()
        self.assertEqual(len(calls), 1)
        self.assertIsNotNone(model.fusion.weight.grad)
        self.assertGreater(torch.count_nonzero(model.correction_joint.weight.grad).item(), 0)
        self.assertIsNone(model.token_table.grad)
        torch.testing.assert_close(model.token_table, before, rtol=0, atol=0)

    @unittest.skipUnless(torch is not None and torch.backends.mps.is_available(),
                         "MPS training device is unavailable")
    def test_real_mps_forward_backward_matches_cpu_equations(self):
        import copy
        cpu = self.model(correction="source_rnn")
        mps = copy.deepcopy(cpu).to("mps")
        inputs = self.inputs()
        prior = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            results = []
            for model, device in ((cpu, "cpu"), (mps, "mps")):
                out = model(*(value.to(device) for value in inputs))
                logits = out.base_states @ model.token_table.T
                logits = logits + out.correction_features @ out.correction_weight.T
                loss = logits.square().mean() + out.confidence.square().mean()
                loss.backward()
                results.append((logits.detach().cpu(), out.confidence.detach().cpu()))
            torch.testing.assert_close(results[0][0], results[1][0], rtol=2e-5, atol=2e-5)
            torch.testing.assert_close(results[0][1], results[1][1], rtol=2e-5, atol=2e-5)
            for (_, c), (_, m) in zip(cpu.named_parameters(), mps.named_parameters()):
                self.assertIsNotNone(c.grad)
                self.assertIsNotNone(m.grad)
                torch.testing.assert_close(c.grad, m.grad.cpu(), rtol=2e-4, atol=2e-5)
        finally:
            torch.use_deterministic_algorithms(prior)


if __name__ == "__main__":
    unittest.main()
