import math
import random

import numpy as np
import pytest
from Bio.Align import PairwiseAligner as BioPairwiseAligner

from mojo_biopython.Align import PairwiseAligner
from mojo_biopython._lib import addr, lib, u8


def configured(cls, mode="global", scores=(2.0, -1.0, -2.0, -0.5)):
    aligner = cls()
    aligner.mode = mode
    (
        aligner.match_score,
        aligner.mismatch_score,
        aligner.open_gap_score,
        aligner.extend_gap_score,
    ) = scores
    return aligner


def test_default_scores_match_biopython():
    mojo = PairwiseAligner()
    bio = BioPairwiseAligner()
    for name in (
        "mode",
        "match_score",
        "mismatch_score",
        "open_gap_score",
        "extend_gap_score",
    ):
        assert getattr(mojo, name) == getattr(bio, name)


@pytest.mark.parametrize("mode", ["global", "local"])
@pytest.mark.parametrize(
    "scores",
    [
        (2.0, -1.0, -2.0, -0.5),
        (1.0, 0.0, -1.0, -1.0),
        (2.0, -3.0, -1.0, -0.2),
        (1.0, -10.0, -1.0, -1.0),
    ],
)
def test_random_scores_match_biopython(mode, scores):
    random.seed(1741)
    mojo = configured(PairwiseAligner, mode, scores)
    bio = configured(BioPairwiseAligner, mode, scores)
    for _ in range(80):
        target = "".join(random.choices("ACGT", k=random.randrange(1, 24)))
        query = "".join(random.choices("ACGT", k=random.randrange(1, 24)))
        assert mojo.score(target, query) == pytest.approx(bio.score(target, query))


def test_global_traceback_matches_unique_biopython_alignment():
    mojo = configured(PairwiseAligner)
    bio = configured(BioPairwiseAligner)
    got = mojo.align("GAACT", "GAT")[0]
    expected = bio.align("GAACT", "GAT")[0]
    assert got.score == expected.score
    assert np.array_equal(got.coordinates, expected.coordinates)
    assert np.array_equal(got.aligned, expected.aligned)


@pytest.mark.parametrize("length", [4, 5, 7, 8, 9])
def test_traceback_simd_block_and_tail_match_biopython(length):
    sequence = ("ACGTTGCAAG" * 2)[:length]
    mojo = configured(PairwiseAligner)
    bio = configured(BioPairwiseAligner)
    got = mojo.align(sequence, sequence)[0]
    expected = bio.align(sequence, sequence)[0]
    assert got.score == expected.score
    assert np.array_equal(got.coordinates, expected.coordinates)


def test_local_traceback_simd_tail_matches_biopython():
    mojo = configured(PairwiseAligner, "local")
    bio = configured(BioPairwiseAligner, "local")
    got = mojo.align("TTACGTTGCAAGTAA", "GGACGTTGCAACC")[0]
    expected = bio.align("TTACGTTGCAAGTAA", "GGACGTTGCAACC")[0]
    assert got.score == expected.score
    assert np.array_equal(got.coordinates, expected.coordinates)


def test_local_traceback_matches_unique_biopython_alignment():
    mojo = configured(PairwiseAligner, "local")
    bio = configured(BioPairwiseAligner, "local")
    got = mojo.align("TTACCGTAA", "GGACCGCC")[0]
    expected = bio.align("TTACCGTAA", "GGACCGCC")[0]
    assert got.score == expected.score
    assert np.array_equal(got.coordinates, expected.coordinates)
    assert np.array_equal(got.aligned, expected.aligned)


def test_local_alignment_has_no_result_without_positive_score():
    aligner = configured(PairwiseAligner, "local")
    aligner.mismatch_score = -10
    assert aligner.score("AAAA", "TTTT") == 0
    assert len(aligner.align("AAAA", "TTTT")) == 0


def test_affine_gap_is_open_plus_extensions():
    aligner = configured(PairwiseAligner)
    assert aligner.score("AAAAA", "AA") == pytest.approx(1.0)


@pytest.mark.parametrize("query_length", [3, 4, 5, 7])
def test_score_simd_block_and_tail_match_biopython(query_length):
    target = "ACGTTGCAAGT"
    query = "ACGTAAC"[:query_length]
    mojo = configured(PairwiseAligner)
    bio = configured(BioPairwiseAligner)
    assert mojo.score(target, query) == pytest.approx(bio.score(target, query))


def test_alignment_collection_and_array_protocol():
    alignments = configured(PairwiseAligner).align("GAACT", "GAT")
    assert len(alignments) == 1
    assert alignments[-1] is alignments[0]
    alignment = alignments[0]
    assert alignment.target == "GAACT"
    assert alignment.query == "GAT"
    assert alignment.shape == (2, 5)
    assert "GAACT" in str(alignment)
    assert np.asarray(alignment).shape == (2, 5)
    assert b"".join(np.asarray(alignment)[1]) == b"GA--T"


def test_score_aliases_update_symmetric_gap_parameters():
    aligner = PairwiseAligner()
    aligner.open_internal_insertion_score = -3
    aligner.extend_internal_deletion_score = -0.25
    assert aligner.open_gap_score == -3
    assert aligner.extend_gap_score == -0.25


def test_reverse_strand_score_matches_biopython():
    mojo = configured(PairwiseAligner)
    bio = configured(BioPairwiseAligner)
    assert mojo.score("AACCGGTT", "AACC", strand="-") == pytest.approx(
        bio.score("AACCGGTT", "AACC", strand="-")
    )


def test_invalid_configuration_is_rejected():
    aligner = PairwiseAligner()
    with pytest.raises(ValueError):
        aligner.mode = "semiglobal"
    aligner.wildcard = "?"
    with pytest.raises(NotImplementedError):
        aligner.score("AC", "AC")
    with pytest.raises(ValueError, match="zero length"):
        PairwiseAligner().score("", "AC")
    aligner = PairwiseAligner()
    aligner.match_score = float("nan")
    with pytest.raises(ValueError, match="finite"):
        aligner.score("AC", "AC")


def test_alignment_ffi_rejects_short_workspace():
    target = u8(b"AC")
    query = u8(b"AC")
    work = np.empty(1, dtype=np.float64)
    score = lib().mbp_alignment_score(
        addr(target), addr(query), 2, 2, 1.0, 0.0, -1.0, -1.0, 0, addr(work), 1
    )
    assert math.isnan(score)


def test_alignment_ffi_rejects_short_trace_buffers():
    target = u8(b"AC")
    query = u8(b"AC")
    scores = np.empty(18, dtype=np.float64)
    traces = np.empty(1, dtype=np.uint8)
    operations = np.empty(5, dtype=np.uint8)
    result = np.empty(5, dtype=np.float64)
    status = lib().mbp_alignment_trace(
        addr(target),
        addr(query),
        2,
        2,
        1.0,
        0.0,
        -1.0,
        -1.0,
        0,
        addr(scores),
        addr(traces),
        addr(operations),
        addr(result),
        scores.size,
        traces.size,
        operations.size,
        result.size,
    )
    assert status == -1
