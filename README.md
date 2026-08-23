# mojo-biopython

`mojo-biopython` is a standalone Mojo port of the compute-heavy core of
[Biopython](https://biopython.org/) sequence alignment, plus byte-level FASTA
and FASTQ scanning. It presents the same class, function, and method names as
the covered Biopython APIs under the `mojo_biopython` package. Biopython is only
a development dependency used for parity tests; the runtime depends on NumPy
and the compiled Mojo shared library.

This is an early, deliberately narrow port with parity tests around the covered
API. On the measured machine it is faster than Biopython 1.87 for alignment
traceback, FASTQ parsing, and reverse complement. Score-only alignment is near
parity, while FASTA parsing remains slower.

## Covered subset

- `Align.PairwiseAligner` global and local alignment with match/mismatch scores,
  symmetric affine open/extend gap scores, linear-memory `score()`, and one
  deterministic optimal traceback from `align()`
- `Alignment.score`, `coordinates`, `aligned`, `shape`, `sequences`, `target`,
  `query`, string formatting, and conversion to a two-row NumPy character array
- `SeqIO.parse()` and `SeqIO.read()` for FASTA, FASTQ, and FASTQ-Sanger,
  including multiline sequences and qualities
- `SeqIO.write()`, `SeqIO.convert()`, and `SeqIO.to_dict()` for the covered
  formats
- lightweight `Seq` and `SeqRecord` objects, record slicing, Phred qualities,
  and IUPAC DNA/RNA `reverse_complement()`

The port does not yet cover substitution matrices, wildcard scoring,
asymmetric insertion/deletion or terminal gap penalties, enumeration of every
equally optimal alignment, codon alignment, MAF/SAM/BAM/GenBank parsers,
features and annotations, translation, or the rest of Biopython. Minus-strand
scoring is supported; minus-strand traceback coordinates are not yet
Biopython-compatible. Traceback uses quadratic memory, while score-only
alignment uses linear memory.

## Install

The repository pins the tested Mojo nightly. Install its environment, build
the shared library, and run the parity suite:

```bash
pixi install
pixi run build
pixi run test
```

The build produces `dist/libmojo-biopython.so`. The Pixi environment adds
`python/` to `PYTHONPATH`, so no editable install is needed.

## Usage

The API differs from upstream only in the top-level package name for the
covered calls:

```python
from io import StringIO

from mojo_biopython import SeqIO
from mojo_biopython.Align import PairwiseAligner

aligner = PairwiseAligner()
aligner.match_score = 2.0
aligner.mismatch_score = -1.0
aligner.open_gap_score = -2.0
aligner.extend_gap_score = -0.5

alignment = aligner.align("GAACT", "GAT")[0]
assert alignment.score == 3.5
assert alignment.coordinates.tolist() == [[0, 2, 4, 5], [0, 2, 2, 3]]

record = SeqIO.read(StringIO(">alpha example\nACGTACGT\n"), "fasta")
assert record.id == "alpha"
assert str(record.seq.reverse_complement()) == "ACGTACGT"
```

The complete example runs with:

```bash
pixi run python examples/quickstart.py
```

## Benchmarks

Measured by an actual `pixi run bench` invocation on the publication checkout
on an Intel Xeon E5-2697 v4 at 2.30 GHz running Linux 6.8.0-136-generic. Times
are the best samples after warmup, as implemented in `bench/bench.py`. Speedup
is `Biopython / Mojo`, so values below 1 mean Mojo is slower. Parsing timings
include creation of the public Python record objects.

| Kernel | Problem | Mojo | Biopython 1.87 | Speedup |
|---|---:|---:|---:|---:|
| global score | 2,500 x 2,500 nt | 30.61 ms | 28.39 ms | 0.93x |
| global traceback | 1,400 x 1,400 nt | 12.28 ms | 18.72 ms | 1.53x |
| FASTA parse | 60k records / 11.5 MB | 230.63 ms | 207.08 ms | 0.90x |
| FASTQ parse | 40k reads / 9.2 MB | 245.49 ms | 286.79 ms | 1.17x |
| reverse complement | 10 million nt | 11.02 ms | 21.71 ms | 1.97x |

No GPU path is included. Parsing and reverse complement are byte-streaming
kernels, while the affine dynamic program performs fewer than two floating
point operations per byte moved and has row dependencies. Their arithmetic
intensity is too low to recover device transfer and launch costs.

## How it works

All native code is in one Mojo compilation unit. Python owns every allocation
and passes contiguous, dtype-specific NumPy buffer addresses through `ctypes`.
Each exported C-ABI function receives explicit input lengths and output
capacities, rejects null or undersized buffers before constructing an
`UnsafePointer`, and returns a checked failure status. Calls are synchronous,
so the Python locals keep every NumPy allocation alive for the entire native
call. No Mojo allocation crosses the FFI boundary.

The affine-gap dynamic program keeps match, deletion, and insertion states.
Both `score()` and `align()` roll six contiguous `float64` rows. Match and
deletion states use native-width SIMD blocks with a scalar remainder; the
left-dependent insertion state is resolved in lane order. Traceback stores
three byte predecessor planes and returns a compact operation stream from which
Python builds Biopython-style coordinates and aligned blocks.

FASTA and FASTQ kernels scan the original contiguous byte buffer and return
only `int64` field boundaries. FASTQ scanning counts sequence and quality
symbols across wrapped lines before Python decodes headers and constructs
records. Reverse complement uses SIMD `uint8` blocks and a scalar remainder
with the full ambiguous IUPAC complement mapping. Inputs of at least 32 million
bases are divided among eight independent CPU tasks; smaller inputs stay serial
to avoid launch overhead.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

The test suite compares alignment scores and tracebacks, parsed record fields,
Phred qualities, sequence transformations, validation behavior, and
serialization directly against Biopython 1.87. It also exercises SIMD
remainders, the parallel threshold, non-contiguous Python buffers, and
undersized native output buffers.
