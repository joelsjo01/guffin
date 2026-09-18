"""Minimal strict GGUF-v3 directory reader for the frozen F16 token table."""

from __future__ import annotations

import hashlib
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path


class GGUFError(ValueError):
    pass


@dataclass(frozen=True)
class TensorEntry:
    name: str
    dimensions: tuple[int, ...]
    ggml_type: int
    relative_offset: int


class _Reader:
    def __init__(self, data: mmap.mmap) -> None:
        self.data = data
        self.offset = 0

    def take(self, count: int) -> bytes:
        if count < 0 or self.offset > len(self.data) or count > len(self.data) - self.offset:
            raise GGUFError("truncated GGUF")
        result = self.data[self.offset:self.offset + count]
        self.offset += count
        return result

    def unpack(self, form: str) -> tuple[object, ...]:
        size = struct.calcsize(form)
        return struct.unpack(form, self.take(size))

    def u32(self) -> int:
        return self.unpack("<I")[0]

    def u64(self) -> int:
        return self.unpack("<Q")[0]

    def string(self) -> str:
        length = self.u64()
        if length > 1 << 30:
            raise GGUFError("GGUF string exceeds bounded size")
        try:
            return self.take(length).decode("utf-8")
        except UnicodeDecodeError as error:
            raise GGUFError("GGUF string is not UTF-8") from error


_SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _skip_value(reader: _Reader, value_type: int, depth: int = 0) -> object | None:
    if depth > 4:
        raise GGUFError("GGUF metadata array nesting is excessive")
    if value_type in _SCALAR_SIZES:
        raw = reader.take(_SCALAR_SIZES[value_type])
        formats = {
            0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
            6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
        }
        return struct.unpack(formats[value_type], raw)[0]
    if value_type == 8:
        return reader.string()
    if value_type == 9:
        element_type, count = reader.u32(), reader.u64()
        if count > 100_000_000:
            raise GGUFError("GGUF metadata array exceeds bounded count")
        for _ in range(count):
            _skip_value(reader, element_type, depth + 1)
        return None
    raise GGUFError(f"unsupported GGUF metadata type {value_type}")


class GGUFFile:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = path.open("rb")
        self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            reader = _Reader(self._map)
            if reader.take(4) != b"GGUF" or reader.u32() != 3:
                raise GGUFError("only GGUF version 3 is supported")
            tensor_count, metadata_count = reader.u64(), reader.u64()
            if tensor_count == 0 or tensor_count > 1_000_000 or metadata_count > 1_000_000:
                raise GGUFError("GGUF directory counts are invalid")
            alignment = 32
            metadata: dict[str, object] = {}
            for _ in range(metadata_count):
                key, value_type = reader.string(), reader.u32()
                value = _skip_value(reader, value_type)
                if value is not None:
                    metadata[key] = value
                if key == "general.alignment":
                    if not isinstance(value, int) or value == 0 or value > 4096 or value & (value - 1):
                        raise GGUFError("GGUF alignment metadata is invalid")
                    alignment = value
            tensors: dict[str, TensorEntry] = {}
            for _ in range(tensor_count):
                name, rank = reader.string(), reader.u32()
                if not name or name in tensors or rank == 0 or rank > 4:
                    raise GGUFError("GGUF tensor identity/rank is invalid")
                dimensions = tuple(reader.u64() for _ in range(rank))
                ggml_type, relative_offset = reader.u32(), reader.u64()
                if any(value == 0 for value in dimensions):
                    raise GGUFError("GGUF tensor has a zero dimension")
                tensors[name] = TensorEntry(name, dimensions, ggml_type, relative_offset)
            self.data_offset = (reader.offset + alignment - 1) & ~(alignment - 1)
            self.tensors = tensors
            self.metadata = metadata
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if hasattr(self, "_map"):
            self._map.close()
            del self._map
        if hasattr(self, "_file"):
            self._file.close()
            del self._file

    def __enter__(self) -> "GGUFFile":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def sha256(self) -> str:
        """Hash the same open mapping used for tensor views.

        Hashing a separate path open would allow a rename between identity
        validation and tensor loading. The bounded view avoids copying GGUF
        payloads while keeping the digest and all later reads on one file
        description.
        """
        digest = hashlib.sha256()
        view = memoryview(self._map)
        try:
            for start in range(0, len(view), 8 * 1024 * 1024):
                digest.update(view[start:start + 8 * 1024 * 1024])
        finally:
            view.release()
        return digest.hexdigest()

    def f16_matrix(self, name: str):
        try:
            entry = self.tensors[name]
        except KeyError as error:
            raise GGUFError(f"GGUF tensor {name!r} is absent") from error
        if entry.ggml_type != 1 or len(entry.dimensions) != 2:
            raise GGUFError(f"GGUF tensor {name!r} is not a two-dimensional F16 tensor")
        byte_count = entry.dimensions[0] * entry.dimensions[1] * 2
        offset = self.data_offset + entry.relative_offset
        if offset > len(self._map) or byte_count > len(self._map) - offset:
            raise GGUFError(f"GGUF tensor {name!r} exceeds the file")
        try:
            import numpy
        except ImportError as error:
            raise RuntimeError("numpy is required for offline training") from error
        # GGUF dimensions are fastest-to-slowest. Matrix rows therefore use
        # the second directory dimension and columns use the first.
        return numpy.ndarray(
            shape=(entry.dimensions[1], entry.dimensions[0]),
            dtype=numpy.float16,
            buffer=self._map,
            offset=offset,
        )

    def floating_vector(self, name: str):
        try:
            entry = self.tensors[name]
        except KeyError as error:
            raise GGUFError(f"GGUF tensor {name!r} is absent") from error
        if len(entry.dimensions) != 1 or entry.ggml_type not in (0, 1):
            raise GGUFError(f"GGUF tensor {name!r} is not an F32/F16 vector")
        try:
            import numpy
        except ImportError as error:
            raise RuntimeError("numpy is required for offline training") from error
        dtype = numpy.float32 if entry.ggml_type == 0 else numpy.float16
        byte_count = entry.dimensions[0] * numpy.dtype(dtype).itemsize
        offset = self.data_offset + entry.relative_offset
        if offset > len(self._map) or byte_count > len(self._map) - offset:
            raise GGUFError(f"GGUF tensor {name!r} exceeds the file")
        return numpy.ndarray(
            shape=(entry.dimensions[0],), dtype=dtype,
            buffer=self._map, offset=offset,
        )
