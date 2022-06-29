import csv
import json
from contextlib import contextmanager
from typing import (Any, Awaitable, Callable, NamedTuple, Optional, Type,
                    TypeVar, Union, cast)

from sqlitedict import SqliteDict

from shared import project_dir

T = TypeVar("T")


class Mol(NamedTuple):
    smiles: str
    name: str
    desc: str
    additional: str
    synthesis: str
    char: str

    def __str__(self):
        return f"{self.smiles}, {self.name}, {self.desc}, {self.additional}, {self.synthesis}, {self.char}"


@contextmanager
def read_csv(filename: str, newline: Optional[str], delimiter: Optional[str]):
    with open(
        f"{project_dir}/main/{filename}", encoding="utf-8", newline=newline
    ) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=delimiter or ",")
        next(csvreader, None)  # header
        yield csvreader


def data():
    with read_csv("data.csv", None, None) as reader:
        return [Mol(*row) for row in reader]
