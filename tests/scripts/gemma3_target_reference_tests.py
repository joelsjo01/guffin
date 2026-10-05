import importlib.util
import math
from pathlib import Path
import struct
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("gemma3_target_reference", ROOT / "scripts/gemma3_target_reference.py")
TARGET = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TARGET)
try:
    import numpy as np
    import torch
except ImportError:
    torch = None


class Gemma3TargetReferenceContractTests(unittest.TestCase):
    def test_complete_target_geometry_is_independent_of_drafter(self):
        shapes = TARGET.expected_shapes()
        self.assertEqual(len(shapes), 626)
        self.assertEqual(shapes["token_embd.weight"], (3840, 262144))
        self.assertEqual(shapes["blk.47.attn_q.weight"], (3840, 4096))
        self.assertEqual(shapes["blk.47.attn_k.weight"], (3840, 2048))
        self.assertEqual(shapes["blk.47.ffn_down.weight"], (15360, 3840))
        self.assertNotIn("output.weight", shapes)

    def test_bad_checkpoint_fails_before_loading_tensors(self):
        artifact = mock.Mock()
        artifact.sha256.return_value = "0" * 64
        with self.assertRaisesRegex(ValueError, "authentication"):
            TARGET.authenticate(artifact)


@unittest.skipIf(torch is None, "offline CPU PyTorch/NumPy reference dependencies are not installed")
class Gemma3TargetReferenceEquationTests(unittest.TestCase):
    def test_q4_k_all_group_scales_minima_and_nibble_planes(self):
        scales = [0, 1, 15, 16, 31, 32, 47, 63]
        minima = [63, 47, 32, 31, 16, 15, 1, 0]
        packed = bytearray(12)
        for group in range(4):
            packed[group] = scales[group] | ((scales[group + 4] >> 4) << 6)
            packed[group + 4] = minima[group] | ((minima[group + 4] >> 4) << 6)
            packed[group + 8] = (scales[group + 4] & 15) | ((minima[group + 4] & 15) << 4)
        quant = [((i * 7 + i // 32) % 16) for i in range(256)]
        packed_quant = bytes(quant[64 * group + i] | (quant[64 * group + 32 + i] << 4)
                             for group in range(4) for i in range(32))
        block = struct.pack("<ee", .125, -.25) + packed + packed_quant
        actual = TARGET.decode_k_quant(block * 4, (512, 2), 12)
        expected = np.array([.125 * scales[i // 32] * quant[i] + .25 * minima[i // 32]
                             for i in range(256)], dtype=np.float32)
        np.testing.assert_array_equal(actual, np.tile(expected, (2, 2)))

    def test_q6_k_signed_scales_all_bit_planes_and_half_blocks(self):
        scales = [-128, -127, -64, -33, -1, 0, 1, 2, 16, 31, 32, 63, 64, 95, 126, 127]
        quant = [(i * 23 + i // 32) % 64 for i in range(256)]
        low, high = bytearray(128), bytearray(64)
        for half in range(2):
            for lane in range(32):
                q = [quant[half * 128 + quarter * 32 + lane] for quarter in range(4)]
                low[half * 64 + lane] = (q[0] & 15) | ((q[2] & 15) << 4)
                low[half * 64 + 32 + lane] = (q[1] & 15) | ((q[3] & 15) << 4)
                high[half * 32 + lane] = sum((q[quarter] >> 4) << (quarter * 2) for quarter in range(4))
        block = low + high + struct.pack("<16be", *scales, -.0625)
        expected = np.array([-.0625 * scales[i // 16] * (quant[i] - 32) for i in range(256)], dtype=np.float32)
        np.testing.assert_array_equal(TARGET.decode_k_quant(block * 2, (256, 2), 14), np.tile(expected, (2, 1)))

    def test_quantized_decoder_rejects_truncation_partial_blocks_and_nonfinite(self):
        for quant_type, size, scale_offset in ((12, 144, 0), (14, 210, 208)):
            with self.assertRaisesRegex(ValueError, "length"):
                TARGET.decode_k_quant(bytes(size - 1), (256, 1), quant_type)
            with self.assertRaisesRegex(ValueError, "blocks"):
                TARGET.decode_k_quant(bytes(size), (255, 1), quant_type)
            invalid = bytearray(size)
            invalid[scale_offset:scale_offset + 2] = struct.pack("<e", float("inf"))
            with self.assertRaisesRegex(ValueError, "non-finite"):
                TARGET.decode_k_quant(invalid, (256, 1), quant_type)
        with self.assertRaisesRegex(ValueError, "Q4_K and Q6_K"):
            TARGET.decode_k_quant(bytes(34), (256, 1), 8)

    def test_chunked_projection_reads_bounded_rows_and_complete_output(self):
        matrix = torch.arange(11 * 256, dtype=torch.float32).reshape(11, 256) / 1000
        x = torch.tensor([[.01] * 256, [-.125] * 256])
        fake = types.SimpleNamespace(tensors={"weight": types.SimpleNamespace(dimensions=(256, 11))})
        with mock.patch.object(TARGET, "tensor_rows", side_effect=lambda g, n, first, count: matrix[first:first + count]) as rows:
            actual = TARGET.project(fake, "weight", x, chunk_rows=3)
        torch.testing.assert_close(actual, x @ matrix.T)
        self.assertEqual([(call.args[2], call.args[3]) for call in rows.call_args_list], [(0, 3), (3, 3), (6, 3), (9, 2)])

    def test_global_rope_factor_eight_applies_at_first_and_high_positions(self):
        x = torch.tensor([[[1., 2., 3., 4.]]] * 4)
        positions = torch.tensor([0, 1, 32768, 131071])
        for local in (False, True):
            actual = TARGET.rope(x, positions, local)
            expected = []
            for p in positions.tolist():
                t = p if local else p / 8
                angles = (t, t / math.sqrt(1e4 if local else 1e6))
                expected.append([[math.cos(angles[0]) - 3 * math.sin(angles[0]),
                    2 * math.cos(angles[1]) - 4 * math.sin(angles[1]),
                    math.sin(angles[0]) + 3 * math.cos(angles[0]),
                    2 * math.sin(angles[1]) + 4 * math.cos(angles[1])]])
            # F32 high-position frequency multiplication has ~1e-4 radian
            # rounding; check against scalar equations at the resulting angle.
            torch.testing.assert_close(actual, torch.tensor(expected), atol=3e-4, rtol=1e-4)
        self.assertFalse(torch.equal(TARGET.rope(x, positions, True)[1], TARGET.rope(x, positions, False)[1]))

    def test_grouped_attention_window_wrap_scale_and_future_mask(self):
        positions = torch.arange(1026)
        q = torch.zeros((1026, 4, 4))
        k = torch.zeros((1026, 2, 4))
        v = torch.ones((1026, 2, 4))
        v[:, 1] *= 3
        v[0, 0] = 4097
        v[1025] = 1e6
        local = TARGET.attention(q, k, v, positions, True, query_chunk=63)
        global_value = TARGET.attention(q, k, v, positions, False, query_chunk=71)
        torch.testing.assert_close(local[1023, :2], torch.full((2, 4), 5.0), atol=1e-5, rtol=0)
        torch.testing.assert_close(local[1024, :2], torch.ones((2, 4)), atol=1e-5, rtol=0)
        torch.testing.assert_close(local[1024, 2:], torch.full((2, 4), 3.0), atol=1e-5, rtol=0)
        torch.testing.assert_close(global_value[1024, :2], torch.full((2, 4), (4097 + 1024) / 1025), atol=1e-5, rtol=0)
        # Nonzero scores prove the 1/sqrt(head_dim) scale, with no softcap.
        tiny_q = torch.tensor([[[0., 0., 0., 0.]], [[2., 0., 0., 0.]]])
        tiny_k = torch.tensor([[[0., 0., 0., 0.]], [[4., 0., 0., 0.]]])
        tiny_v = torch.tensor([[[0., 0., 0., 0.]], [[1., 1., 1., 1.]]])
        actual = TARGET.attention(tiny_q, tiny_k, tiny_v, torch.arange(2), False)
        torch.testing.assert_close(actual[1], torch.full((1, 4), 1 / (1 + math.exp(-4))), atol=1e-7, rtol=0)

    def test_f16_storage_rounds_current_rows_before_f32_attention(self):
        # Independent binary16 bits at ties and range boundaries authenticate
        # the offline oracle instead of inheriting an engine conversion helper.
        original = [1 + 2**-11, 1 + 3 * 2**-11, -1 - 2**-11,
                    2**-25, 3 * 2**-25, 65504., 65519., 65520.]
        expected_bits = [0x3c00, 0x3c02, 0xbc00, 0, 2, 0x7bff, 0x7bff, 0x7c00]
        rounded = torch.tensor(original, dtype=torch.float32).to(torch.float16)
        self.assertEqual([int(x) & 0xffff for x in rounded.view(torch.int16)], expected_bits)
        q = torch.tensor([[[1.25, -2., .25, 0.]], [[2., 1., -.5, .25]]])
        k = torch.tensor([[[.3126, -.7213, .1942, .9131]], [[-.3492, .5111, .8753, -.1246]]])
        v = torch.tensor([[[1.0006, -1.0006, .00031, 1230.9257]], [[-.1246, 3.0012, .00073, -549.2043]]])
        rounded_k, rounded_v = k.half(), v.half()
        actual = TARGET.attention(q, rounded_k, rounded_v, torch.arange(2), False, query_chunk=1)
        expected = []
        for position in range(2):
            scores = [sum(float(q[position, 0, i]) * float(rounded_k[j, 0, i]) for i in range(4)) / 2
                      for j in range(position + 1)]
            mass = [math.exp(x - max(scores)) for x in scores]
            expected.append([[sum(mass[j] * float(rounded_v[j, 0, i]) for j in range(position + 1)) / sum(mass)
                              for i in range(4)]])
        torch.testing.assert_close(actual, torch.tensor(expected), atol=1e-5, rtol=2e-7)
        self.assertTrue(torch.equal(actual[0], rounded_v[0].float()))
        self.assertFalse(torch.equal(actual[0], v[0]))
        torch.testing.assert_close(TARGET.attention(q, rounded_k, rounded_v, torch.arange(2), False, query_chunk=2),
                                   actual, atol=1e-5, rtol=2e-7)

    def test_invalid_inputs_and_preserved_output_fail_before_checkpoint_access(self):
        for tokens, selected in (([], [0]), ([TARGET.VOCAB], [0]), ([2], []), ([2], [1])):
            with self.assertRaisesRegex(ValueError, "invalid"):
                TARGET.reference(None, tokens, selected, lambda *args: None)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing"
            output.write_text("preserve")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                TARGET.generate(Path("missing.gguf"), output)
            self.assertEqual(output.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
