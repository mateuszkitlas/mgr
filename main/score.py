from typing import Optional, Tuple, TypedDict

from shared import Fn


class DictScore(TypedDict):
    sa: float
    sc: float
    ra: float
    mf: float
    mfgb: float
    syba: float


class Score:
    def __init__(
        self, sa: float, sc: float, ra: float, mf: float, mfgb: float, syba: float
    ):
        self.sa = sa
        self.sc = sc
        self.ra = ra
        self.mf = mf
        self.mfgb = mfgb
        self.syba = syba

    @staticmethod
    def from_json(j: DictScore):
        return Score(**j)

    @staticmethod
    def getters() -> list[Tuple[str, Fn["Score", float]]]:
        return [
            ("sa", lambda s: s.sa),
            ("sc", lambda s: s.sc),
            ("ra", lambda s: s.ra),
            ("mf", lambda s: s.mf),
            ("mfgb", lambda s: s.mfgb),
            ("syba", lambda s: s.syba),
        ]

    def as_dict(self) -> DictScore:
        return vars(self)


class DictSmiles(TypedDict):
    smiles: str
    score: DictScore
    transforms: Optional[int]


class Smiles:
    def __init__(self, smiles: str, score: Score, transforms: Optional[int]):
        self.smiles = smiles
        self.score = score
        self.transforms = transforms

    def as_dict(self) -> DictSmiles:
        return vars(self)

    @staticmethod
    def from_json(j: DictSmiles):
        return Smiles(j["smiles"], Score.from_json(j["score"]), j["transforms"])
