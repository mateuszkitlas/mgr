from typing import Tuple, TypeVar, TypedDict

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
    expandable_smileses: list[str]
    children: list["AiTree"]
