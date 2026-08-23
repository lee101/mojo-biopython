"""FASTA and FASTQ parsing with Biopython-compatible entry points."""

from __future__ import annotations

import os
from collections.abc import Iterable

import numpy as np

from .Seq import Seq
from .SeqRecord import SeqRecord
from ._lib import addr, lib, u8

_PHRED_TABLE = bytes((value - 33) & 0xFF for value in range(256))


def _read(handle) -> bytes:
    if isinstance(handle, (str, bytes, os.PathLike)):
        with open(handle, "rb") as stream:
            return stream.read()
    data = handle.read()
    return data.encode() if isinstance(data, str) else bytes(data)


def _header_fields(raw: bytes) -> tuple[str, str]:
    description = raw.decode("utf-8")
    identifier = description.split(None, 1)[0] if description else ""
    return identifier, description


def _clean_lines(block: bytes) -> bytes:
    trimmed = block.rstrip(b"\r\n")
    return trimmed if b"\n" not in trimmed else block.translate(None, b"\r\n")


def _parse_fasta(data: bytes):
    if not data:
        return
    if not data.startswith(b">"):
        raise ValueError("FASTA content must start with '>'")
    array = u8(data)
    count = data.count(b"\n>") + 1
    positions = np.empty((count, 4), dtype=np.int64)
    found = int(
        lib().mbp_fasta_scan(
            addr(array, dtype=np.uint8),
            len(array),
            addr(positions, dtype=np.int64, writable=True),
            positions.shape[0],
        )
    )
    if found < 0:
        raise RuntimeError("Mojo FASTA scanner rejected its buffers")
    make_seq = Seq._from_ascii_bytes
    make_record = SeqRecord._from_parsed
    for header_start, header_end, seq_start, seq_end in positions[:found].tolist():
        description = data[header_start:header_end].decode("utf-8")
        identifier = description.split(None, 1)[0] if description else ""
        sequence = data[seq_start:seq_end].translate(None, b"\r\n \t")
        yield make_record(make_seq(sequence), identifier, description)


def _parse_fastq(data: bytes):
    if not data:
        return
    array = u8(data)
    line_count = data.count(b"\n") + int(not data.endswith(b"\n"))
    capacity = max(1, (line_count + 3) // 4)
    positions = np.empty((capacity, 6), dtype=np.int64)
    found = int(
        lib().mbp_fastq_scan(
            addr(array, dtype=np.uint8),
            len(array),
            addr(positions, dtype=np.int64, writable=True),
            positions.shape[0],
        )
    )
    if found == -2:
        raise ValueError("Whitespace is not allowed in the sequence")
    if found == -3:
        raise ValueError("Invalid character in quality string")
    if found < 0:
        raise ValueError("Invalid FASTQ structure or sequence/quality length mismatch")
    values = iter(positions[:found].ravel().tolist())
    for hstart, hend, sstart, send, qstart, qend in zip(
        values, values, values, values, values, values
    ):
        identifier, description = _header_fields(data[hstart:hend])
        sequence = _clean_lines(data[sstart:send])
        plus_caption = data[send + 1 : qstart].rstrip(b"\r\n")
        if plus_caption and plus_caption != data[hstart:hend]:
            raise ValueError("Sequence and quality captions differ")
        quality = _clean_lines(data[qstart:qend])
        scores = list(quality.translate(_PHRED_TABLE))
        record = SeqRecord(
            Seq(sequence), id=identifier, name=identifier, description=description
        )
        record.letter_annotations["phred_quality"] = scores
        yield record


def parse(handle, format, alphabet=None):
    """Turn a FASTA or FASTQ file into an iterator of ``SeqRecord`` objects."""
    if alphabet is not None:
        raise ValueError("The alphabet argument is no longer supported")
    normalized = format.lower()
    if normalized in {"fasta", "fa", "fna"}:
        return _parse_fasta(_read(handle))
    if normalized in {"fastq", "fastq-sanger"}:
        return _parse_fastq(_read(handle))
    raise ValueError(f"Unknown format {format!r}; supported formats are fasta and fastq")


def read(handle, format, alphabet=None):
    """Read exactly one record from a FASTA or FASTQ file."""
    iterator = parse(handle, format, alphabet)
    try:
        record = next(iterator)
    except StopIteration:
        raise ValueError("No records found in handle") from None
    try:
        next(iterator)
    except StopIteration:
        return record
    raise ValueError("More than one record found in handle")


def _format_record(record: SeqRecord, format: str) -> str:
    normalized = format.lower()
    description = record.description
    title = description if description.startswith(record.id) else (
        record.id if description == "<unknown description>" else f"{record.id} {description}"
    )
    if normalized in {"fasta", "fa", "fna"}:
        sequence = str(record.seq)
        lines = [sequence[index : index + 60] for index in range(0, len(sequence), 60)]
        return f">{title}\n" + "\n".join(lines) + "\n"
    if normalized in {"fastq", "fastq-sanger"}:
        try:
            quality = record.letter_annotations["phred_quality"]
        except KeyError:
            raise ValueError("No suitable quality scores found in letter_annotations") from None
        if len(quality) != len(record):
            raise ValueError("Record sequence and quality scores must have equal length")
        encoded_scores = []
        for score in quality:
            if isinstance(score, (bool, np.bool_)):
                value = int(score)
            elif isinstance(score, (int, np.integer)):
                value = int(score)
            else:
                raise TypeError("Phred quality scores must be integers")
            if not 0 <= value <= 93:
                raise ValueError("Phred quality scores must be between 0 and 93")
            encoded_scores.append(chr(value + 33))
        encoded = "".join(encoded_scores)
        return f"@{title}\n{record.seq}\n+\n{encoded}\n"
    raise ValueError(f"Unknown format {format!r}; supported formats are fasta and fastq")


def write(sequences, handle, format) -> int:
    """Write records and return the number written."""
    records = [sequences] if isinstance(sequences, SeqRecord) else sequences
    close = isinstance(handle, (str, bytes, os.PathLike))
    stream = open(handle, "w", encoding="utf-8") if close else handle
    count = 0
    try:
        for record in records:
            stream.write(_format_record(record, format))
            count += 1
    finally:
        if close:
            stream.close()
    return count


def to_dict(sequences: Iterable[SeqRecord], key_function=None) -> dict[str, SeqRecord]:
    result = {}
    for record in sequences:
        key = record.id if key_function is None else key_function(record)
        if key in result:
            raise ValueError(f"Duplicate key {key!r}")
        result[key] = record
    return result


def convert(in_file, in_format, out_file, out_format) -> int:
    return write(parse(in_file, in_format), out_file, out_format)
