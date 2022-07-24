from typing import Optional, Tuple, TypedDict

from shared import Db, Fn


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
        "sa, sc, ra, mf, mfgb, syba"
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


class DictAiSmiles(TypedDict):
    smiles: str
    transforms: Optional[int]


class AiSmiles:
    def __init__(self, smiles: str, score: Score, transforms: Optional[int]):
        self.smiles = smiles
        self.score = score
        self.transforms = transforms

    def as_dict(self) -> DictAiSmiles:
        return {"smiles": self.smiles, "transforms": self.transforms}

    @staticmethod
    def from_json(j: DictAiSmiles, db: Db):
        return AiSmiles(
            j["smiles"],
            Score.from_json(db.read(j["smiles"], DictScore)),
            j["transforms"],
        )
