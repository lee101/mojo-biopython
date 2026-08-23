"""The subset of Bio.SeqRecord used by FASTA and FASTQ workflows."""

from __future__ import annotations

from .Seq import Seq


class SeqRecord:
    __slots__ = (
        "seq",
        "id",
        "name",
        "description",
        "dbxrefs",
        "features",
        "annotations",
        "letter_annotations",
    )

    def __init__(
        self,
        seq,
        id="<unknown id>",
        name="<unknown name>",
        description="<unknown description>",
        dbxrefs=None,
        features=None,
        annotations=None,
        letter_annotations=None,
    ):
        self.seq = seq if isinstance(seq, Seq) else Seq(seq)
        self.id = id
        self.name = name
        self.description = description
        self.dbxrefs = [] if dbxrefs is None else list(dbxrefs)
        self.features = [] if features is None else list(features)
        self.annotations = {} if annotations is None else dict(annotations)
        self.letter_annotations = (
            {} if letter_annotations is None else dict(letter_annotations)
        )

    @classmethod
    def _from_parsed(cls, seq: Seq, identifier: str, description: str) -> "SeqRecord":
        record = cls.__new__(cls)
        record.seq = seq
        record.id = identifier
        record.name = identifier
        record.description = description
        record.dbxrefs = []
        record.features = []
        record.annotations = {}
        record.letter_annotations = {}
        return record

    def __len__(self) -> int:
        return len(self.seq)

    def __getitem__(self, key):
        if isinstance(key, slice):
            result = SeqRecord(
                self.seq[key],
                id=self.id,
                name=self.name,
                description=self.description,
                annotations=self.annotations,
            )
            result.letter_annotations = {
                name: values[key] for name, values in self.letter_annotations.items()
            }
            return result
        return self.seq[key]

    def __repr__(self) -> str:
        return (
            f"SeqRecord(seq={self.seq!r}, id={self.id!r}, "
            f"name={self.name!r}, description={self.description!r})"
        )

    def format(self, format: str) -> str:
        from .SeqIO import _format_record

        return _format_record(self, format)
