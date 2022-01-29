from typing import Optional, Tuple, TypedDict, TypeVar

T = TypeVar("T")
Timed = Tuple[float, T]


class RawScore(TypedDict):
    sa: Timed[float]
    sc: Timed[float]
    ra: Timed[float]
    mf: Timed[float]
    syba: Timed[float]


class AiTree(TypedDict):
    is_solved: bool
    score: float
    expandable_smiles: list[str]
    in_stock_smiles: list[str]
    children: list[Optional["AiTree"]]
