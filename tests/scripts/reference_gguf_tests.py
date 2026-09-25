import hashlib
from pathlib import Path
import struct
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from reference_gguf import GGUFError, GGUFFile


def gguf_fixture():
    payload = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 1, 0))
    payload += struct.pack("<Q", 1) + b"x"
    payload += struct.pack("<IQIQ", 1, 1, 1, 0)
    payload += bytes((-len(payload)) % 32)
    payload += b"\x00\x00"
    return bytes(payload)


class ReferenceGGUFTests(unittest.TestCase):
    def test_gguf_digest_uses_the_same_open_mapping_as_tensor_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "model.gguf"
            original = gguf_fixture()
            path.write_bytes(original)
            with GGUFFile(path) as gguf:
                replacement = directory / "replacement.gguf"
                replacement.write_bytes(b"replacement")
                replacement.replace(path)
                self.assertEqual(gguf.sha256(), hashlib.sha256(original).hexdigest())
                self.assertEqual(gguf.tensors["x"].dimensions, (1,))
                self.assertEqual(gguf.tensors["x"].ggml_type, 1)

    def test_rejects_invalid_and_truncated_directories(self):
        payload = gguf_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.gguf"
            for invalid in (b"FAIL" + payload[4:], payload[:8], payload[:24], payload[:40]):
                with self.subTest(size=len(invalid)):
                    path.write_bytes(invalid)
                    with self.assertRaises(GGUFError):
                        GGUFFile(path)


if __name__ == "__main__":
    unittest.main()
