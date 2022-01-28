from typing import Callable, Generic, Tuple, TypeVar

from .types import RawScore

T = TypeVar("T")
U = TypeVar("U")
R = TypeVar("R")


class Score(Generic[T]):
    def __init__(self, sa: T, sc: T, ra: T, syba: T):
        self.sa = sa
        self.sc = sc
        self.ra = ra
        self.syba = syba

    def map_with(self, score: "Score[U]", fn: Callable[[T, U], R]) -> "Score[R]":
        return Score[R](
            fn(self.sa, score.sa),
            fn(self.sc, score.sc),
            fn(self.ra, score.ra),
            fn(self.syba, score.syba),
        )

    def map(self, fn: Callable[[T], R]) -> "Score[R]":
        return Score[R](fn(self.sa), fn(self.sc), fn(self.ra), fn(self.syba))

    def __str__(self):
        return f"sa: {self.sa}, sc: {self.sc}, ra: {self.ra}, syba: {self.syba}"

    @staticmethod
    def from_raw(r: RawScore) -> Tuple["Score[float]", "Score[float]"]:
        return (
            Score(sa=r["sa"][0], sc=r["sc"][0], ra=r["ra"][0], syba=r["syba"][0]),
            Score(sa=r["sa"][1], sc=r["sc"][1], ra=r["ra"][1], syba=r["syba"][1]),
        )

    def to_list(self):
        return [self.sa, self.sc, self.ra, self.syba]

    def add(self, s: "Score[T]", f: Callable[[T, T], T]) -> "Score[T]":
        return self.map_with(s, f)


class Smiles:
    def __init__(self, smiles: str, score: Score[float]):
        self.smiles = smiles
        self.score = score

    def __str__(self):
        return f"{self.score}, smiles: {self.smiles}"
