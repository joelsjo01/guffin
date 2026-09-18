import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.export_recurrent import TENSORS, checked_json, pack_weights, publish

try:
    import torch
except ModuleNotFoundError:
    torch = None


class SpeculationExportTests(unittest.TestCase):
    def test_strict_bounded_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"receipt.json"
            for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'[]'):
                path.write_bytes(raw)
                with self.assertRaises(ValueError):
                    checked_json(path)
            path.write_text(json.dumps({"schema":"test"}))
            self.assertEqual(checked_json(path)[1], {"schema":"test"})
            with self.assertRaises(ValueError):
                checked_json(path, maximum=4)

    def test_immutable_atomic_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"weights.lgweights"
            publish(path, b"original")
            with self.assertRaises(FileExistsError):
                publish(path, b"replacement")
            self.assertEqual(path.read_bytes(), b"original")
            self.assertEqual(list(path.parent.iterdir()), [path])

    @unittest.skipIf(torch is None, "offline training dependencies unavailable")
    def test_exact_master_weights_and_complete_directory(self):
        shapes = ((4,8),(4,),(4,8),(4,8),(4,),(4,8),(4,),(8,4),(1,4),(1,))
        state = {name:torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)/7
                 for name,shape in zip(TENSORS,shapes)}
        directory, payload = pack_weights(state,4,8)
        self.assertEqual((directory,payload), pack_weights(state,4,8))
        offset = 0
        for index,(name,shape) in enumerate(zip(TENSORS,shapes)):
            rows,columns,at,size = struct.unpack_from("<IIQQ",directory,index*56)
            self.assertEqual((rows,columns), shape if len(shape)==2 else (1,shape[0]))
            self.assertEqual(at,offset)
            raw = state[name].numpy().astype("<f4").tobytes()
            self.assertEqual(payload[at:at+size],raw)
            self.assertEqual(directory[index*56+24:(index+1)*56],hashlib.sha256(raw).digest())
            offset += size
        self.assertEqual(offset,len(payload))
        for kind in ("missing","extra","shape","dtype","nan"):
            bad = dict(state)
            if kind=="missing":
                del bad[TENSORS[-1]]
            elif kind=="extra":
                bad["unexpected"] = torch.zeros(1)
            elif kind=="shape":
                bad[TENSORS[0]] = torch.zeros(8,4)
            elif kind=="dtype":
                bad[TENSORS[0]] = bad[TENSORS[0]].half()
            else:
                bad[TENSORS[0]] = torch.full((4,8),float("nan"))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                pack_weights(bad,4,8)


if __name__ == "__main__":
    unittest.main()
