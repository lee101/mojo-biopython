"""Load the Mojo shared library and declare its C ABI."""

from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LIBRARY = ROOT / "dist" / "libmojo-biopython.so"

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mbp_alignment_score": ([I, I, I, I, F, F, F, F, I, I, I], F),
    "mbp_alignment_trace": (
        [I, I, I, I, F, F, F, F, I, I, I, I, I, I, I, I, I],
        I,
    ),
    "mbp_fasta_scan": ([I, I, I, I], I),
    "mbp_fastq_scan": ([I, I, I, I], I),
    "mbp_reverse_complement": ([I, I, I, I], I),
}

_library: ctypes.CDLL | None = None


def build() -> Path:
    subprocess.run(["bash", str(ROOT / "build" / "build.sh")], cwd=ROOT, check=True)
    return LIBRARY


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        if not LIBRARY.exists():
            build()
        _library = ctypes.CDLL(str(LIBRARY))
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def u8(data: bytes | bytearray | memoryview | str) -> np.ndarray:
    if isinstance(data, str):
        data = data.encode("ascii")
    try:
        array = np.frombuffer(data, dtype=np.uint8)
    except (BufferError, TypeError, ValueError):
        array = np.frombuffer(bytes(data), dtype=np.uint8)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    return array


def addr(
    array: np.ndarray, *, dtype: np.dtype | type | None = None, writable: bool = False
) -> int:
    if not isinstance(array, np.ndarray) or not array.flags.c_contiguous:
        raise TypeError("FFI buffers must be C-contiguous NumPy arrays")
    if dtype is not None and array.dtype != np.dtype(dtype):
        raise TypeError(f"FFI buffer must have dtype {np.dtype(dtype)}")
    if writable and not array.flags.writeable:
        raise TypeError("FFI output buffers must be writable")
    if array.nbytes and not array.ctypes.data:
        raise ValueError("FFI buffer has a null data pointer")
    return int(array.ctypes.data)


if __name__ == "__main__":
    print(build())
