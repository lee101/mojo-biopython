"""A compact immutable sequence type and sequence-level functions."""

from __future__ import annotations

import numpy as np

from ._lib import addr, lib, u8


class Seq:
    __slots__ = ("_data",)

    def __init__(self, data: str | bytes | bytearray | memoryview | "Seq"):
        if isinstance(data, Seq):
            self._data = data._data
        elif isinstance(data, str):
            data.encode("ascii")
            self._data = data
        else:
            self._data = bytes(data).decode("ascii")

    @classmethod
    def _from_ascii_bytes(cls, data: bytes) -> "Seq":
        sequence = cls.__new__(cls)
        sequence._data = data.decode("ascii")
        return sequence

    def __str__(self) -> str:
        return self._data

    def __repr__(self) -> str:
        shown = self._data if len(self._data) <= 60 else self._data[:54] + "..."
        return f"Seq({shown!r})"

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key):
        value = self._data[key]
        return Seq(value) if isinstance(key, slice) else value

    def __bytes__(self) -> bytes:
        return self._data.encode("ascii")

    def __eq__(self, other) -> bool:
        if isinstance(other, Seq):
            return self._data == other._data
        if isinstance(other, str):
            return self._data == other
        return NotImplemented

    def __add__(self, other) -> "Seq":
        return Seq(self._data + str(other))

    def upper(self) -> "Seq":
        return Seq(self._data.upper())

    def lower(self) -> "Seq":
        return Seq(self._data.lower())

    def complement(self) -> "Seq":
        return Seq(reverse_complement(self._data)[::-1])

    def reverse_complement(self, inplace: bool = False) -> "Seq":
        if inplace:
            raise TypeError("Sequence is immutable")
        return Seq(reverse_complement(self._data))


def reverse_complement(sequence, inplace: bool = False):
    """Return the reverse complement, preserving the input's common type."""
    if inplace:
        raise TypeError("Sequence is immutable")
    is_seq = isinstance(sequence, Seq)
    is_bytes = isinstance(sequence, (bytes, bytearray))
    raw = bytes(sequence) if is_seq else (
        bytes(sequence) if is_bytes else str(sequence).encode("ascii")
    )
    source = u8(raw)
    destination = np.empty(len(source), dtype=np.uint8)
    status = lib().mbp_reverse_complement(
        addr(source, dtype=np.uint8),
        len(source),
        addr(destination, dtype=np.uint8, writable=True),
        destination.size,
    )
    if status != 0:
        raise RuntimeError("Mojo reverse-complement kernel rejected its buffers")
    result = destination.tobytes()
    if is_seq:
        return Seq(result)
    if is_bytes:
        return result
    return result.decode("ascii")
