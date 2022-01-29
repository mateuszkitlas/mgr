from typing import Callable, Generic, Tuple, TypedDict, TypeVar

from .utils import serialize_dict

from .types import RawScore

T = TypeVar("T")
U = TypeVar("U")
R = TypeVar("R")


class JsonScoreFloat(TypedDict):
    sa: float
    sc: float
    ra: float
    mf: float
    syba: float


class Score(Generic[T]):
    def __init__(self, sa: T, sc: T, ra: T, mf: T, syba: T):
        self.sa = sa
        self.sc = sc
        self.ra = ra
        self.mf = mf
        self.syba = syba

    def map_with(self, score: "Score[U]", fn: Callable[[T, U], R]) -> "Score[R]":
        return Score[R](
            sa=fn(self.sa, score.sa),
            sc=fn(self.sc, score.sc),
            ra=fn(self.ra, score.ra),
            mf=fn(self.mf, score.mf),
            syba=fn(self.syba, score.syba),
        )

    def map(self, fn: Callable[[T], R]) -> "Score[R]":
        return Score[R](
            sa=fn(self.sa),
            sc=fn(self.sc),
            ra=fn(self.ra),
            syba=fn(self.syba),
            mf=fn(self.mf),
        )

    def __str__(self):
        return serialize_dict(self.json(), ",")

    @staticmethod
    def from_raw(r: RawScore) -> Tuple["Score[float]", "Score[float]"]:
        return (
            Score(
                sa=r["sa"][0],
                sc=r["sc"][0],
                ra=r["ra"][0],
                syba=r["syba"][0],
                mf=r["mf"][0],
            ),
            Score(
                sa=r["sa"][1],
                sc=r["sc"][1],
                ra=r["ra"][1],
                syba=r["syba"][1],
                mf=r["mf"][1],
            ),
        )

    @staticmethod
    def from_json(j: JsonScoreFloat):
        return Score(sa=j["sa"], sc=j["sc"], ra=j["ra"], syba=j["syba"], mf=j["mf"])

    def to_list(self):
        return [self.sa, self.sc, self.ra, self.mf, self.syba]

    def add(self, s: "Score[T]", f: Callable[[T, T], T]) -> "Score[T]":
        return self.map_with(s, f)

    @staticmethod
    def getters() -> list[Tuple[str, Callable[["Score[U]"], U]]]:
        return [
            ("sa", lambda s: s.sa),
            ("sc", lambda s: s.sc),
            ("ra", lambda s: s.ra),
            ("mf", lambda s: s.mf),
            ("syba", lambda s: s.syba),
        ]

    def json(self) -> JsonScoreFloat:
        assert (
            isinstance(self.sa, float)
            and isinstance(self.sc, float)
            and isinstance(self.ra, float)
            and isinstance(self.mf, float)
            and isinstance(self.syba, float)
        )
        return {
            "sa": self.sa,
            "sc": self.sc,
            "ra": self.ra,
            "mf": self.mf,
            "syba": self.syba,
        }


class JsonSmiles(TypedDict):
    smiles: str
    score: JsonScoreFloat


class Smiles:
    def __init__(self, smiles: str, score: Score[float]):
        self.smiles = smiles
        self.score = score

    def __str__(self):
        return f"{self.score}, smiles: {self.smiles}"

    def json(self) -> JsonSmiles:
        return {"smiles": self.smiles, "score": self.score.json()}

    @staticmethod
    def from_json(j: JsonSmiles):
        return Smiles(j["smiles"], Score.from_json(j["score"]))
