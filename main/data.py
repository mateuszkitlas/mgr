import csv
import json
import os
from contextlib import contextmanager
from typing import NamedTuple, Optional, Tuple

from shared import paracetamol_smiles, project_dir

from .tree import JsonTree
from .types import AiInput, AiTree


class Mol(NamedTuple):
    smiles: str
    name: str
    desc: str
    additional: str
    synthesis: str
    char: str

    def __str__(self):
        return f"{self.smiles}, {self.name}, {self.desc}, {self.additional}, {self.synthesis}, {self.char}"


paracetamol = Mol(paracetamol_smiles, "paracetamol", "", "", "", "")


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


def save_trees(data: list[Tuple[AiInput, AiTree]], filename: str):
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    with open(os.path.join(results_dir, filename), "w") as f:
        json.dump(data, f)


def load_trees(filename: str, retries: int = 3) -> list[Tuple[AiInput, AiTree]]:
    try:
        with open(os.path.join(results_dir, filename)) as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as e:
        # when `./run main` and `./run stats` concurently, this error may raise
        if retries > 0:
            return load_trees(filename, retries - 1)
        else:
            raise e
