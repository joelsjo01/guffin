import importlib.util
from pathlib import Path
import struct
import hashlib
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("convert_vicuna_draft", ROOT / "scripts/convert_vicuna_draft.py")
CONVERTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERTER)


class VicunaConversionTests(unittest.TestCase):
    def test_complete_checkpoint_geometry(self):
        shapes = CONVERTER.expected_shapes()
        self.assertEqual(len(shapes), 21)
        self.assertEqual(sum(__import__("math").prod(s) for s in shapes.values()), 68030208)
        self.assertEqual(shapes["model.layers.1.mlp.down_proj.weight"], (768, 3072))
        self.assertEqual(shapes["lm_head.weight"], (32000, 768))

    def test_protobuf_rejects_truncation_and_invalid_wire_types(self):
        for raw in (b"\x80", b"\x00", b"\x0a\x02x", b"\x0d\0", b"\x0b", b"\xff" * 12):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                list(CONVERTER.protobuf_fields(raw))
        self.assertEqual(list(CONVERTER.protobuf_fields(b"\x08\x02\x12\x01x")), [(1, 0, 2), (2, 2, b"x")])

    def test_source_authentication_and_immutable_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "draft.gguf"
            output.write_bytes(b"preserved")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                CONVERTER.convert(root, output)
            self.assertEqual(output.read_bytes(), b"preserved")
            (root / "config.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "authentication"):
                CONVERTER.convert(root, root / "new.gguf")
            self.assertFalse((root / "new.gguf").exists())

    def test_metadata_binary_shapes(self):
        self.assertEqual(CONVERTER.metadata_value(4, 768), struct.pack("<I", 768))
        self.assertEqual(CONVERTER.metadata_value(9, (5, [1, 6])), struct.pack("<IQii", 5, 2, 1, 6))

    def test_full_pinned_conversion_when_local_checkpoint_available(self):
        source = ROOT / "models/vicuna_68m"
        if not (source / "model.safetensors").exists():
            return
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "draft.gguf"
            digest = CONVERTER.convert(source, path)
            self.assertEqual(digest, "2038020ca7b9a2584252f4bf1cfd5e4cb833130b0853822e58cfa538299211b7")
            self.assertEqual(path.stat().st_size, 136789856)
            with path.open("rb") as stream:
                self.assertEqual(hashlib.file_digest(stream, "sha256").hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
