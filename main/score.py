from typing import TypeVar, Generic, Type, Callable
from .types import RawScore, Timed

T = TypeVar("T")
U = TypeVar("U")
R = TypeVar("R")


class Score(Generic[T]):
    def __init__(self, sa: T, sc: T, ra: T, syba: T):
        self.sa = sa
        self.sc = sc
        self.ra = ra
        self.syba = syba

    def map_with(self, r: Type[R], score: "Score[U]", fn: Callable[[T, U], R]) -> "Score[R]":
        return Score[R](
            fn(self.sa, score.sa),
            fn(self.sc, score.sc),
            fn(self.ra, score.ra),
            fn(self.syba, score.syba),
        )

    def map(self, fn: Callable[[T], R]) -> "Score[R]":
        return Score[R](fn(self.sa), fn(self.sc), fn(self.ra), fn(self.syba))

    def __str__(self):
        return str(self.json())

    def json(self):
        return vars(self)

    @staticmethod
    def from_raw(raw: RawScore) -> "Score[Timed[float]]":
        return Score(sa=raw["sa"], sc=raw["sc"], ra=raw["ra"], syba=raw["syba"])
