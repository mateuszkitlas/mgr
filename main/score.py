from typing import TypeVar, Generic
T = TypeVar("T")
class Score(Generic[T]):
    def __init__(self, sa, sc, ra, syba):
        self.sa = sa
        self.sc = sc
        self.ra = ra
        self.syba = syba
    def map_with(self, r, score, fn):
        return Score(
            fn(self.sa, score.sa),
            fn(self.sc, score.sc),
            fn(self.ra, score.ra),
            fn(self.syba, score.syba),
        )
    def map(self, fn):
        return Score(
            fn(self.sa),
            fn(self.sc),
            fn(self.ra),
            fn(self.syba),
        )
    def __str__(self): return str(self.json())
    def json(self): return vars(self)
    @staticmethod
    def from_raw(raw):
        return Score(sa=raw["sa"], sc=raw["sc"], ra=raw["ra"], syba=raw["syba"])
