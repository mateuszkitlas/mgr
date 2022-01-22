import csv
import json
from typing import Any, List


class Mol:
    def __init__(self, smiles: str, name: str, desc: str, additional: str, synthesis: str, char: str):
        self.smiles = smiles
        self.name = name
        self.desc = desc
        self.additional = additional
        self.synthesis = synthesis
        self.char = char

    @staticmethod
    def from_csv_row(row: List[str]):
        return Mol(row[0], row[1], row[2], row[3], row[4], row[5])


def read_csv(fp: str, small: bool):
    with open(fp, encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader, None)  # header
        all = [Mol.from_csv_row(row) for row in csvreader]
        return all[:5] if small else all


def save(name: str, data: Any):
    with open(name, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


class Saver:
    def __init__(self, name: str):
        self.name = name
        self.data: List[Any] = []

    def append(self, data: Any):
        self.data.append(data)
        save(self.name, self.data)
