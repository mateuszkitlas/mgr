from typing import Optional, Tuple, TypeVar, TypedDict

T = TypeVar("T")
Timed = Tuple[float, T]

class RawScore(TypedDict):
    sa: Timed[float]
    sc: Timed[float]
    ra: Timed[float]
    syba: Timed[float]

class AiTree(TypedDict):
    is_solved: bool
    score: float
    expandable_smiles: list[str]
    in_stock_smiles: list[str]
    children: list[Optional["AiTree"]]
