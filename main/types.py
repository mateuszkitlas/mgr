from typing import Optional, Tuple, TypedDict, TypeVar

T = TypeVar("T")
Timed = Tuple[float, T]


class AiTree(TypedDict):
    is_solved: bool
    ai_score: float
    expandable: list[Tuple[str, Optional[int]]]
    in_stock: list[Tuple[str, Optional[int]]]
    children: list[Optional["AiTree"]]
