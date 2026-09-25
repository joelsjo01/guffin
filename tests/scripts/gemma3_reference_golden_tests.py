import importlib.util
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("gemma3_reference_golden", ROOT / "scripts/gemma3_reference_golden.py")
GOLDEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GOLDEN)
try:
    import numpy as np
    import torch
except ImportError:
    torch = None


class Gemma3ReferenceContractTests(unittest.TestCase):
    def test_complete_real_checkpoint_geometry(self):
        shapes = GOLDEN.expected_shapes()
        self.assertEqual(len(shapes), 236)
        self.assertEqual(sum(math.prod(shape) for shape in shapes.values()), 268098176)
        self.assertEqual(shapes["token_embd.weight"], (640, 262144))
        self.assertNotIn("output.weight", shapes)
        self.assertEqual(shapes["blk.17.attn_q_norm.weight"], (256,))
        self.assertEqual(len(GOLDEN.WRAP_IDS), 536)
        self.assertTrue(all(0 <= token < GOLDEN.VOCAB for token in GOLDEN.WRAP_IDS))

    def test_authentication_fails_before_reading_tensors(self):
        artifact = mock.Mock()
        artifact.sha256.return_value = "0" * 64
        with self.assertRaisesRegex(ValueError, "authentication"):
            GOLDEN.authenticate(artifact)


@unittest.skipIf(torch is None, "offline CPU PyTorch/NumPy golden dependencies are not installed")
class Gemma3ReferenceEquationTests(unittest.TestCase):
    def test_q8_signed_values_scale_and_matrix_order(self):
        integers = [-128, -127, -1, 0, 1, 127] + [7] * 26
        payload = struct.pack("<e32b", .125, *integers) + struct.pack("<e32b", -.25, *integers)
        actual = GOLDEN.decode_q8(payload, (32, 2))
        expected = np.array([[value * .125 for value in integers], [value * -.25 for value in integers]], dtype=np.float32)
        np.testing.assert_array_equal(actual, expected)
        for bad in (payload[:-1], payload + b"x"):
            with self.assertRaisesRegex(ValueError, "length"):
                GOLDEN.decode_q8(bad, (32, 2))
        with self.assertRaisesRegex(ValueError, "blocks"):
            GOLDEN.decode_q8(payload, (31, 2))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            GOLDEN.decode_q8(struct.pack("<e32b", float("inf"), *integers), (32, 1))

    def test_gguf_norm_does_not_apply_hf_offset_twice(self):
        x = torch.tensor([[1.0, -2.0, 3.0, -4.0]])
        gamma = torch.tensor([.25, .5, 1.0, 2.0])
        expected = [value * weight / math.sqrt(7.5 + GOLDEN.EPSILON) for value, weight in zip(x[0].tolist(), gamma.tolist())]
        torch.testing.assert_close(GOLDEN.rms(x, gamma), torch.tensor([expected]), atol=1e-7, rtol=1e-7)
        scale = torch.tensor(640, dtype=torch.float32).sqrt()
        self.assertNotEqual(scale.item(), scale.bfloat16().float().item())

    def test_neox_rope_pairs_halves_not_adjacent_rows(self):
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        actual = GOLDEN.rope(x, torch.tensor([7]), 10000.0)
        angles = (7.0, .07)
        expected = [math.cos(angles[0]) - 3 * math.sin(angles[0]),
                    2 * math.cos(angles[1]) - 4 * math.sin(angles[1]),
                    math.sin(angles[0]) + 3 * math.cos(angles[0]),
                    2 * math.sin(angles[1]) + 4 * math.cos(angles[1])]
        torch.testing.assert_close(actual, torch.tensor([[expected]]), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(GOLDEN.rope(x, torch.tensor([0]), 1e6), x, atol=0, rtol=0)

    def test_causal_local_window_first_wrap_and_global_distinction(self):
        positions = torch.arange(536)
        local, global_mask = GOLDEN.causal_mask(positions, True), GOLDEN.causal_mask(positions, False)
        self.assertEqual(local[511].sum().item(), 512)
        self.assertTrue(local[511, 0].item())
        self.assertFalse(local[512, 0].item())
        self.assertTrue(local[512, 1].item())
        self.assertTrue(local[512, 512].item())
        self.assertFalse(local[512, 513].item())
        self.assertFalse(local[535, 23].item())
        self.assertTrue(local[535, 24].item())
        self.assertEqual(local[535].sum().item(), 512)
        self.assertEqual(global_mask[535].sum().item(), 536)

    def test_tanh_gelu_matches_scalar_equation(self):
        x = torch.tensor([-4.0, -.8, 0.0, .5, 3.0])
        expected = [.5 * value * (1 + math.tanh(math.sqrt(2 / math.pi) * (value + .044715 * value ** 3))) for value in x.tolist()]
        torch.testing.assert_close(GOLDEN.gelu_tanh(x), torch.tensor(expected), rtol=1e-5, atol=2e-7)

    def test_production_cache_keeps_f32_threshold_information(self):
        # Actual layer-0 checkpoint values on either side of one F16 midpoint.
        # F32 storage must not magnify a ~1e-5 projection difference into .03125.
        values = torch.tensor([-41.109371185302734, -41.10938262939453], dtype=torch.float32)
        actual = GOLDEN.cache_values(values, "f32")
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, values, atol=0, rtol=0)
        diagnostic = GOLDEN.cache_values(values, "f16")
        self.assertEqual(diagnostic.dtype, torch.float16)
        self.assertEqual(abs(float(diagnostic[0]) - float(diagnostic[1])), .03125)
        with self.assertRaisesRegex(ValueError, "cache dtype"):
            GOLDEN.cache_values(values, "bf16")

    def test_invalid_input_and_existing_output_fail_without_model_access(self):
        for tokens, positions in (([], [0]), ([GOLDEN.VOCAB], [0]), ([2], [1]), ([2], [])):
            with self.assertRaisesRegex(ValueError, "invalid"):
                GOLDEN.reference(None, tokens, positions, lambda *args: None)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "preserved"
            output.write_bytes(b"user output")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                GOLDEN.generate(Path("does-not-exist.gguf"), output)
            self.assertEqual(output.read_bytes(), b"user output")


if __name__ == "__main__":
    unittest.main()
