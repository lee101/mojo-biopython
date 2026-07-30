from io import StringIO

import numpy as np
import pytest
from Bio import SeqIO as BioSeqIO
from Bio.Seq import reverse_complement as bio_reverse_complement

from mojo_biopython import Seq, SeqIO, SeqRecord, reverse_complement
from mojo_biopython._lib import addr, lib, u8


FASTA = """>alpha first record
ACGTAC
GT
>beta
NNNNacgt
"""

FASTQ = """@read1 instrument 7
ACGT
AC
+
IIJJ
KL
@read2
NNTA
+
!#5?
"""


def snapshot(records):
    return [
        (
            record.id,
            record.name,
            record.description,
            str(record.seq),
            dict(record.letter_annotations),
        )
        for record in records
    ]


def test_fasta_parse_matches_biopython():
    got = snapshot(SeqIO.parse(StringIO(FASTA), "fasta"))
    expected = snapshot(BioSeqIO.parse(StringIO(FASTA), "fasta"))
    assert got == expected


def test_fastq_parse_matches_biopython_including_wrapping():
    got = snapshot(SeqIO.parse(StringIO(FASTQ), "fastq"))
    expected = snapshot(BioSeqIO.parse(StringIO(FASTQ), "fastq"))
    assert got == expected


def test_parse_from_path(tmp_path):
    path = tmp_path / "records.fa"
    path.write_text(FASTA)
    assert [record.id for record in SeqIO.parse(path, "fasta")] == ["alpha", "beta"]


@pytest.mark.parametrize("format", ["fa", "fna"])
def test_fasta_format_aliases(format):
    assert SeqIO.read(StringIO(">x\nAC\n"), format).id == "x"


def test_fastq_sanger_alias():
    assert SeqIO.read(StringIO("@x\nAC\n+\nII\n"), "fastq-sanger").id == "x"


def test_read_exactly_one_record():
    record = SeqIO.read(StringIO(">only description\nACGT\n"), "fasta")
    assert record.id == "only"
    assert str(record.seq) == "ACGT"
    with pytest.raises(ValueError, match="More than one"):
        SeqIO.read(StringIO(FASTA), "fasta")
    with pytest.raises(ValueError, match="No records"):
        SeqIO.read(StringIO(""), "fasta")


def test_fasta_removes_spaces_and_tabs_like_biopython():
    text = ">spaced\nAC GT\tNN\n"
    assert str(SeqIO.read(StringIO(text), "fasta").seq) == "ACGTNN"


def test_fastq_validation_matches_common_biopython_errors():
    with pytest.raises(ValueError, match="length mismatch"):
        list(SeqIO.parse(StringIO("@x\nACG\n+\nII\n"), "fastq"))
    with pytest.raises(ValueError, match="captions differ"):
        list(SeqIO.parse(StringIO("@x\nAC\n+y\nII\n"), "fastq"))
    with pytest.raises(ValueError, match="Whitespace"):
        list(SeqIO.parse(StringIO("@x\nA C\n+\nIII\n"), "fastq"))
    with pytest.raises(ValueError, match="Invalid character"):
        list(SeqIO.parse(StringIO("@x\nAC\n+\nI\x7f\n"), "fastq"))
    with pytest.raises(ValueError, match="Invalid FASTQ"):
        list(SeqIO.parse(StringIO("@x\n"), "fastq"))


def test_fasta_requires_header_at_start_like_biopython():
    with pytest.raises(ValueError):
        list(SeqIO.parse(StringIO("\n>x\nAC\n"), "fasta"))


def test_fasta_write_matches_biopython(tmp_path):
    record = SeqRecord(Seq("ACGT" * 20), id="x", name="x", description="x long")
    ours = tmp_path / "ours.fa"
    theirs = tmp_path / "theirs.fa"
    assert SeqIO.write([record], ours, "fasta") == 1
    bio_record = next(BioSeqIO.parse(StringIO(ours.read_text()), "fasta"))
    BioSeqIO.write([bio_record], theirs, "fasta")
    assert ours.read_text() == theirs.read_text()


def test_fastq_roundtrip_preserves_quality(tmp_path):
    source = list(SeqIO.parse(StringIO(FASTQ), "fastq"))
    path = tmp_path / "reads.fastq"
    assert SeqIO.write(source, path, "fastq") == 2
    assert snapshot(SeqIO.parse(path, "fastq")) == snapshot(source)


@pytest.mark.parametrize("quality", [[1.5], [-1], [94]])
def test_fastq_write_rejects_narrowing_or_out_of_range_quality(quality):
    record = SeqRecord("A", id="x", letter_annotations={"phred_quality": quality})
    with pytest.raises((TypeError, ValueError)):
        record.format("fastq")


def test_to_dict_and_duplicate_detection():
    records = list(SeqIO.parse(StringIO(FASTA), "fasta"))
    assert list(SeqIO.to_dict(records)) == ["alpha", "beta"]
    with pytest.raises(ValueError, match="Duplicate"):
        SeqIO.to_dict([records[0], records[0]])


def test_convert_fasta_to_fastq_requires_quality(tmp_path):
    source = tmp_path / "source.fa"
    destination = tmp_path / "destination.fastq"
    source.write_text(">x\nAC\n")
    with pytest.raises(ValueError, match="quality"):
        SeqIO.convert(source, "fasta", destination, "fastq")


def test_convert_covered_format(tmp_path):
    source = tmp_path / "source.fa"
    destination = tmp_path / "destination.fa"
    source.write_text(">x\nAC\n")
    assert SeqIO.convert(source, "fasta", destination, "fasta") == 1
    assert destination.read_text() == ">x\nAC\n"


@pytest.mark.parametrize(
    "sequence",
    ["ACGTRYKMSWBDHVN", "acgtrykmswbdhvn", "AUGCUU", ""],
)
def test_reverse_complement_matches_biopython(sequence):
    assert reverse_complement(sequence) == str(bio_reverse_complement(sequence))
    assert str(Seq(sequence).reverse_complement()) == str(
        bio_reverse_complement(sequence)
    )


@pytest.mark.parametrize("length", [31, 32, 33, 65])
def test_reverse_complement_simd_tail(length):
    sequence = ("ACGTRYKMSWBDHVN" * 5)[:length]
    assert reverse_complement(sequence) == str(bio_reverse_complement(sequence))


def test_reverse_complement_parallel_threshold():
    sequence = ("ACGTRYKMSWBDHVN" * 2_133_335)[:32_000_017]
    assert reverse_complement(sequence) == str(bio_reverse_complement(sequence))


def test_ffi_buffer_guards_and_noncontiguous_input():
    sliced = memoryview(b"A_C_G_T")[::2]
    assert u8(sliced).tobytes() == b"ACGT"
    with pytest.raises(TypeError, match="contiguous"):
        addr(np.arange(8, dtype=np.uint8)[::2])
    with pytest.raises(TypeError, match="dtype"):
        addr(np.arange(8, dtype=np.int16), dtype=np.uint8)

    source = u8(b"AC")
    destination = np.empty(1, dtype=np.uint8)
    assert (
        lib().mbp_reverse_complement(
            addr(source), source.size, addr(destination), destination.size
        )
        == -1
    )


def test_scanner_ffi_rejects_short_output_buffer():
    data = u8(b">a\nA\n>b\nC\n")
    positions = np.empty((1, 4), dtype=np.int64)
    assert (
        lib().mbp_fasta_scan(addr(data), data.size, addr(positions), positions.shape[0])
        == -1
    )


def test_seq_and_seqrecord_slicing():
    record = SeqRecord(
        Seq("ACGT"), id="x", description="x", letter_annotations={"phred_quality": [1, 2, 3, 4]}
    )
    sliced = record[1:3]
    assert str(sliced.seq) == "CG"
    assert sliced.letter_annotations["phred_quality"] == [2, 3]


def test_seq_covered_operations():
    sequence = Seq("AcGt")
    assert str(sequence.upper()) == "ACGT"
    assert str(sequence.lower()) == "acgt"
    assert str(sequence.complement()) == "TgCa"
    assert str(sequence + "NN") == "AcGtNN"
    assert bytes(sequence) == b"AcGt"
