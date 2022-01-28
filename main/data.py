import csv
import json
import os
from contextlib import contextmanager
from typing import NamedTuple, Optional, Tuple

from shared import disable_syba, project_dir

from .score import Score, Smiles
from .tree import JsonTree


class Mol(NamedTuple):
    smiles: str
    name: str
    desc: str
    additional: str
    synthesis: str
    char: str

    def __str__(self):
        return f"{self.smiles}, {self.name}, {self.desc}, {self.additional}, {self.synthesis}, {self.char}"


paracetamol = Mol("CC(=O)Nc1ccc(O)cc1", "paracetamol", "", "", "", "")


@contextmanager
def read_csv(filename: str, newline: Optional[str], delimiter: Optional[str]):
    with open(
        os.path.join(project_dir, "main", filename), encoding="utf-8", newline=newline
    ) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=delimiter or ",")
        next(csvreader, None)  # header
        yield csvreader


def data():
    with read_csv("data.csv", None, None) as reader:
        return [Mol(*row) for row in reader]


results_dir = os.path.join(project_dir, "results")


def save_trees(data: list[Tuple[str, JsonTree]], filename: str):
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    with open(os.path.join(results_dir, filename), "w") as f:
        json.dump(data, f)


def load_trees(filename: str) -> list[Tuple[str, JsonTree]]:
    with open(os.path.join(results_dir, filename)) as f:
        return json.load(f)


def test_data():
    _disable_syba = disable_syba()
    # header: SMILES  SAscore SCScore SYBA    RAscore
    with read_csv(f"test.csv", newline="\r\n", delimiter="\t") as reader:
        return [
            Smiles(
                row[0],
                Score(
                    sa=float(row[1]),
                    sc=float(row[2]),
                    ra=float(row[4]),
                    syba=0.0 if _disable_syba else float(row[3]),
                ),
            )
            for row in reader
        ]
