import csv
import json
import os
from contextlib import contextmanager
from typing import Any, List, NamedTuple, Optional

from shared import DISABLE_SYBA, project_dir

from .score import Score, Smiles

paracetamol = "CC(=O)Nc1ccc(O)cc1"


class Mol(NamedTuple):
    smiles: str
    name: str
    desc: str
    additional: str
    synthesis: str
    char: str


@contextmanager
def read_csv(filename: str, newline: Optional[str], delimiter: Optional[str]):
    with open(
        os.path.join(project_dir, "main", filename), encoding="utf-8", newline=newline
    ) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=delimiter)
        next(csvreader, None)  # header
        yield csvreader


def data(small: bool):
    if small:
        return [Mol(paracetamol, "paracetamol", "", "", "", "")]
    with read_csv("data.csv", None, None) as reader:
        return [Mol(*row) for row in reader]


def test_data():
    # header: SMILES  SAscore SCScore SYBA    RAscore
    with read_csv(f"test.csv", newline="\r\n", delimiter="\t") as reader:
        return [
            Smiles(
                row[0],
                Score(
                    float(row[1]),
                    float(row[2]),
                    float(row[4]),
                    0.0 if DISABLE_SYBA else float(row[3]),
                ),
            )
            for row in reader
        ]


def save(name: str, data: Any):
    with open(name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


class Saver:
    def __init__(self, name: str):
        self.name = name
        self.data: List[Any] = []

    def append(self, data: Any):
        self.data.append(data)
        save(self.name, self.data)
