from io import StringIO

from mojo_biopython import SeqIO
from mojo_biopython.Align import PairwiseAligner

aligner = PairwiseAligner()
aligner.match_score = 2.0
aligner.mismatch_score = -1.0
aligner.open_gap_score = -2.0
aligner.extend_gap_score = -0.5

alignment = aligner.align("GAACT", "GAT")[0]
print(f"score: {alignment.score}")
print(alignment)

records = SeqIO.parse(StringIO(">alpha example\nACGTACGT\n"), "fasta")
record = next(records)
print(record.id, record.seq.reverse_complement())
