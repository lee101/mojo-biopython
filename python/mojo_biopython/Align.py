"""Pairwise alignment with a Biopython-shaped Python interface."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import numpy as np

from .Seq import Seq
from ._lib import addr, lib, u8


def _sequence_text(sequence) -> str:
    if isinstance(sequence, Seq):
        return str(sequence)
    if isinstance(sequence, bytes):
        return sequence.decode("ascii")
    return str(sequence)


class Alignment:
    def __init__(
        self,
        target: str,
        query: str,
        score: float,
        operations: np.ndarray,
        start: tuple[int, int],
        end: tuple[int, int],
    ):
        self.sequences = [target, query]
        self.score = float(score)
        self._operations = np.asarray(operations, dtype=np.uint8)
        self._start = start
        self._end = end
        self.coordinates = self._make_coordinates()

    def _make_coordinates(self) -> np.ndarray:
        target, query = self._start
        points = [[target, query]]
        if self._operations.size:
            previous = int(self._operations[0])
            for operation in self._operations:
                operation = int(operation)
                if operation != previous:
                    points.append([target, query])
                    previous = operation
                if operation != 2:
                    target += 1
                if operation != 1:
                    query += 1
            points.append([target, query])
        return np.asarray(points, dtype=np.int64).T

    @property
    def shape(self) -> tuple[int, int]:
        return (2, int(self._operations.size))

    @property
    def aligned(self) -> np.ndarray:
        blocks = []
        target, query = self._start
        block_start = None
        for operation in self._operations:
            operation = int(operation)
            if operation == 0 and block_start is None:
                block_start = (target, query)
            elif operation != 0 and block_start is not None:
                blocks.append((block_start, (target, query)))
                block_start = None
            if operation != 2:
                target += 1
            if operation != 1:
                query += 1
        if block_start is not None:
            blocks.append((block_start, (target, query)))
        result = np.empty((2, len(blocks), 2), dtype=np.int64)
        for index, (begin, finish) in enumerate(blocks):
            result[0, index] = (begin[0], finish[0])
            result[1, index] = (begin[1], finish[1])
        return result

    def _gapped(self) -> tuple[str, str, str]:
        target_text, query_text = self.sequences
        ti, qi = self._start
        target = []
        query = []
        middle = []
        for operation in self._operations:
            if operation == 0:
                tc, qc = target_text[ti], query_text[qi]
                target.append(tc)
                query.append(qc)
                middle.append("|" if tc == qc else ".")
                ti += 1
                qi += 1
            elif operation == 1:
                target.append(target_text[ti])
                query.append("-")
                middle.append("-")
                ti += 1
            else:
                target.append("-")
                query.append(query_text[qi])
                middle.append("-")
                qi += 1
        return "".join(target), "".join(middle), "".join(query)

    def __array__(self, dtype=None, copy=None):
        target, _, query = self._gapped()
        result = np.stack(
            [
                np.frombuffer(target.encode("ascii"), dtype="S1"),
                np.frombuffer(query.encode("ascii"), dtype="S1"),
            ]
        )
        return result.astype(dtype) if dtype is not None else result

    def __str__(self) -> str:
        target, middle, query = self._gapped()
        ti, qi = self._start
        return (
            f"target {ti:>12} {target} {self._end[0]}\n"
            f"{'':19}{middle}\n"
            f"query  {qi:>12} {query} {self._end[1]}\n"
        )

    @property
    def target(self):
        return self.sequences[0]

    @property
    def query(self):
        return self.sequences[1]

    def __repr__(self) -> str:
        return f"<Alignment object (2 rows x {self.shape[1]} columns)>"


class PairwiseAlignments(Sequence[Alignment]):
    def __init__(self, alignment: Alignment | None):
        self._alignment = alignment

    def __len__(self) -> int:
        return int(self._alignment is not None)

    def __getitem__(self, index):
        if isinstance(index, slice):
            values = [] if self._alignment is None else [self._alignment]
            return values[index]
        if self._alignment is not None and index in (0, -1):
            return self._alignment
        raise IndexError("alignment index out of range")

    def __iter__(self) -> Iterator[Alignment]:
        if self._alignment is not None:
            yield self._alignment


class PairwiseAligner:
    """Global or local pairwise aligner with symmetric affine gap scores.

    The public scoring attributes and ``align``/``score`` signatures match
    ``Bio.Align.PairwiseAligner`` for the covered subset.
    """

    def __init__(self, scoring=None, **kwargs):
        if scoring is not None:
            raise ValueError("named scoring schemes are not supported")
        self._mode = "global"
        self.match_score = 1.0
        self.mismatch_score = 0.0
        self._open_gap_score = -1.0
        self._extend_gap_score = -1.0
        self.wildcard = None
        self.epsilon = 1e-6
        self.substitution_matrix = None
        for name, value in kwargs.items():
            if not hasattr(self, name):
                raise AttributeError(f"'PairwiseAligner' object has no attribute {name!r}")
            setattr(self, name, value)

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str):
        if value not in {"global", "local"}:
            raise ValueError("mode must be 'global' or 'local'")
        self._mode = value

    @property
    def open_gap_score(self) -> float:
        return self._open_gap_score

    @open_gap_score.setter
    def open_gap_score(self, value):
        self._open_gap_score = float(value)

    @property
    def extend_gap_score(self) -> float:
        return self._extend_gap_score

    @extend_gap_score.setter
    def extend_gap_score(self, value):
        self._extend_gap_score = float(value)

    @property
    def gap_score(self) -> float | None:
        if self.open_gap_score == self.extend_gap_score:
            return self.open_gap_score
        return None

    @gap_score.setter
    def gap_score(self, value):
        self.open_gap_score = value
        self.extend_gap_score = value

    @property
    def algorithm(self) -> str:
        prefix = "Smith-Waterman" if self.mode == "local" else "Gotoh global"
        return f"{prefix} alignment algorithm"

    def _inputs(self, seqA, seqB, strand: str):
        if strand not in {"+", "-"}:
            raise ValueError("strand must be '+' or '-'")
        target = _sequence_text(seqA)
        query = _sequence_text(seqB)
        if not target or not query:
            raise ValueError("sequence has zero length")
        if strand == "-":
            from .Seq import reverse_complement

            query = reverse_complement(query)
        target_raw = u8(target)
        query_raw = u8(query)
        return target, query, target_raw, query_raw

    def _validate(self):
        if self.wildcard is not None:
            raise NotImplementedError("wildcard scoring is not implemented")
        if self.substitution_matrix is not None:
            raise NotImplementedError("substitution matrices are not implemented")
        for name in (
            "match_score",
            "mismatch_score",
            "open_gap_score",
            "extend_gap_score",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")

    def score(self, seqA, seqB, strand="+") -> float:
        self._validate()
        _, _, target, query = self._inputs(seqA, seqB, strand)
        work = np.empty(6 * (len(query) + 1), dtype=np.float64)
        value = float(
            lib().mbp_alignment_score(
                addr(target, dtype=np.uint8),
                addr(query, dtype=np.uint8),
                len(target),
                len(query),
                self.match_score,
                self.mismatch_score,
                self.open_gap_score,
                self.extend_gap_score,
                self.mode == "local",
                addr(work, dtype=np.float64, writable=True),
                work.size,
            )
        )
        if math.isnan(value):
            raise RuntimeError("Mojo alignment score kernel rejected its buffers")
        return value

    def align(self, seqA, seqB, strand="+") -> PairwiseAlignments:
        self._validate()
        target_text, query_text, target, query = self._inputs(seqA, seqB, strand)
        cells = (len(target) + 1) * (len(query) + 1)
        scores = np.empty(6 * (len(query) + 1), dtype=np.float64)
        traces = np.empty(3 * cells, dtype=np.uint8)
        operations = np.empty(len(target) + len(query) + 1, dtype=np.uint8)
        result = np.empty(5, dtype=np.float64)
        count = lib().mbp_alignment_trace(
            addr(target, dtype=np.uint8),
            addr(query, dtype=np.uint8),
            len(target),
            len(query),
            self.match_score,
            self.mismatch_score,
            self.open_gap_score,
            self.extend_gap_score,
            self.mode == "local",
            addr(scores, dtype=np.float64, writable=True),
            addr(traces, dtype=np.uint8, writable=True),
            addr(operations, dtype=np.uint8, writable=True),
            addr(result, dtype=np.float64, writable=True),
            scores.size,
            traces.size,
            operations.size,
            result.size,
        )
        if count < 0:
            raise RuntimeError("Mojo alignment traceback kernel rejected its buffers")
        if self.mode == "local" and result[0] <= 0.0:
            return PairwiseAlignments(None)
        forward = operations[:count][::-1].copy()
        alignment = Alignment(
            target_text,
            query_text,
            result[0],
            forward,
            (int(result[3]), int(result[4])),
            (int(result[1]), int(result[2])),
        )
        return PairwiseAlignments(alignment)

    def __str__(self) -> str:
        return (
            "Pairwise sequence aligner with parameters\n"
            f"  match_score: {self.match_score:.6f}\n"
            f"  mismatch_score: {self.mismatch_score:.6f}\n"
            f"  open_gap_score: {self.open_gap_score:.6f}\n"
            f"  extend_gap_score: {self.extend_gap_score:.6f}\n"
            f"  mode: {self.mode}\n"
        )


def _score_alias(attribute):
    def getter(self):
        return getattr(self, attribute)

    def setter(self, value):
        setattr(self, attribute, value)

    return property(getter, setter)


for _name in (
    "open_internal_insertion_score",
    "open_left_insertion_score",
    "open_right_insertion_score",
    "open_internal_deletion_score",
    "open_left_deletion_score",
    "open_right_deletion_score",
    "target_open_gap_score",
    "query_open_gap_score",
    "target_internal_open_gap_score",
    "query_internal_open_gap_score",
    "target_left_open_gap_score",
    "query_left_open_gap_score",
    "target_right_open_gap_score",
    "query_right_open_gap_score",
):
    setattr(PairwiseAligner, _name, _score_alias("open_gap_score"))

for _name in (
    "extend_internal_insertion_score",
    "extend_left_insertion_score",
    "extend_right_insertion_score",
    "extend_internal_deletion_score",
    "extend_left_deletion_score",
    "extend_right_deletion_score",
    "target_extend_gap_score",
    "query_extend_gap_score",
    "target_internal_extend_gap_score",
    "query_internal_extend_gap_score",
    "target_left_extend_gap_score",
    "query_left_extend_gap_score",
    "target_right_extend_gap_score",
    "query_right_extend_gap_score",
):
    setattr(PairwiseAligner, _name, _score_alias("extend_gap_score"))
