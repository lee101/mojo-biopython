"""mojo-biopython versus Biopython 1.87 on identical inputs."""

from __future__ import annotations

import io
import os
import platform
import sys
import time

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

from Bio import SeqIO as BioSeqIO  # noqa: E402
from Bio.Align import PairwiseAligner as BioPairwiseAligner  # noqa: E402
from Bio.Seq import reverse_complement as bio_reverse_complement  # noqa: E402

from mojo_biopython import SeqIO, reverse_complement  # noqa: E402
from mojo_biopython.Align import PairwiseAligner  # noqa: E402


def best_time(function, repetitions=3):
    function()
    best = float("inf")
    result = None
    for _ in range(repetitions):
        started = time.perf_counter()
        result = function()
        best = min(best, time.perf_counter() - started)
    return best, result


def configure(cls):
    aligner = cls()
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -2.0
    aligner.extend_gap_score = -0.5
    return aligner


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="ascii") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main():
    mojo = configure(PairwiseAligner)
    bio = configure(BioPairwiseAligner)
    target = ("ACGTTGCAAGTC" * 210)[:2500]
    query = ("ACGTCGCAAGGC" * 210)[:2500]

    rows = []

    mojo_time, mojo_score = best_time(lambda: mojo.score(target, query), 5)
    bio_time, bio_score = best_time(lambda: bio.score(target, query), 5)
    assert mojo_score == bio_score
    rows.append(("global score", "2,500 x 2,500 nt", mojo_time, bio_time))

    trace_target = target[:1400]
    trace_query = query[:1400]

    def mojo_trace():
        alignment = mojo.align(trace_target, trace_query)[0]
        return alignment.score, alignment.coordinates.shape

    def bio_trace():
        alignment = bio.align(trace_target, trace_query)[0]
        return alignment.score, alignment.coordinates.shape

    mojo_time, mojo_result = best_time(mojo_trace, 3)
    bio_time, bio_result = best_time(bio_trace, 3)
    assert mojo_result == bio_result
    rows.append(("global traceback", "1,400 x 1,400 nt", mojo_time, bio_time))

    fasta = "".join(
        f">record_{index} generated sequence\n"
        f"{'ACGT' * 20}\n{'TGCA' * 20}\n"
        for index in range(60_000)
    )

    def mojo_fasta():
        return sum(len(record) for record in SeqIO.parse(io.StringIO(fasta), "fasta"))

    def bio_fasta():
        return sum(
            len(record) for record in BioSeqIO.parse(io.StringIO(fasta), "fasta")
        )

    mojo_time, mojo_result = best_time(mojo_fasta)
    bio_time, bio_result = best_time(bio_fasta)
    assert mojo_result == bio_result
    rows.append(("FASTA parse", "60k records / 11.5 MB", mojo_time, bio_time))

    fastq = "".join(
        f"@read_{index} generated\n"
        f"{'ACGT' * 25}\n+\n{'I' * 100}\n"
        for index in range(40_000)
    )

    def mojo_fastq():
        return sum(
            sum(record.letter_annotations["phred_quality"])
            for record in SeqIO.parse(io.StringIO(fastq), "fastq")
        )

    def bio_fastq():
        return sum(
            sum(record.letter_annotations["phred_quality"])
            for record in BioSeqIO.parse(io.StringIO(fastq), "fastq")
        )

    mojo_time, mojo_result = best_time(mojo_fastq)
    bio_time, bio_result = best_time(bio_fastq)
    assert mojo_result == bio_result
    rows.append(("FASTQ parse", "40k reads / 9.2 MB", mojo_time, bio_time))

    sequence = ("ACGTRYKMSWBDHVN" * 700_000)[:10_000_000]
    mojo_time, mojo_result = best_time(lambda: reverse_complement(sequence), 5)
    bio_time, bio_result = best_time(lambda: str(bio_reverse_complement(sequence)), 5)
    assert mojo_result == bio_result
    rows.append(("reverse complement", "10 million nt", mojo_time, bio_time))

    print(f"Machine: {cpu_name()}; {platform.system()} {platform.release()}")
    print()
    print("| Kernel | Problem | Mojo | Biopython | Speedup |")
    print("|---|---:|---:|---:|---:|")
    for name, problem, mojo_time, bio_time in rows:
        print(
            f"| {name} | {problem} | {mojo_time * 1000:.2f} ms | "
            f"{bio_time * 1000:.2f} ms | {bio_time / mojo_time:.2f}x |"
        )


if __name__ == "__main__":
    main()
