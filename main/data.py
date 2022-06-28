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

@contextmanager
def db_and_mols():
    with Db() as db:
        mols = data()
        count = len(mols)
        for i, mol in enumerate(mols):
            yield db, mol, i, count

class Db:
    def __init__(self):
        self.db = SqliteDict(
            f"{project_dir}/results/db.sqlite", outer_stack=False, autocommit=True
        )

    def _write(self, raw_key: str, value: T) -> T:
        self.db[raw_key] = json.dumps(value, sort_keys=True)
        return value

    async def maybe_create(self, key: Any, f: Callable[[], Awaitable[Any]]):
        raw_key = json.dumps(key, sort_keys=True)
        if raw_key not in self.db:
            self._write(raw_key, await f())

    async def read_or_create(
        self, key: Any, f: Callable[[], Awaitable[T]]
    ) -> Awaitable[T]:
        raw_key = json.dumps(key, sort_keys=True)
        return (
            self.db[raw_key] if raw_key in self.db else self._write(raw_key, await f())
        )

    def read(self, key: Any, type: Type[T]) -> Union[T, None]:
        return cast(type, json.loads(self.db.get(json.dumps(key, sort_keys=True), "null")))

    def write(self, key: Any, value: Any):
        self._write(json.dumps(key, sort_keys=True), value)

    def __enter__(self):
        self.db.__enter__()
        return self

    def __exit__(self, *exc_info):
        self.db.close()
