from typing import Generic, Optional, Tuple, TypedDict, TypeVar

from shared import Fn

from .types import RawScore
from .utils import serialize_dict

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

    @staticmethod
    def getters() -> list[Tuple[str, Fn["Score[U]", U]]]:
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
    transforms: Optional[int]


class Smiles:
    def __init__(self, smiles: str, score: Score[float], transforms: Optional[int]):
        self.smiles = smiles
        self.score = score
        self.transforms = transforms

    def __str__(self):
        return serialize_dict(self.json(), ", ")

    def json(self) -> JsonSmiles:
        return {"smiles": self.smiles, "score": self.score.json(), "transforms": self.transforms}

    @staticmethod
    def from_json(j: JsonSmiles):
        return Smiles(j["smiles"], Score.from_json(j["score"]), j["transforms"])
