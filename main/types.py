from typing import Literal, Optional, Tuple, TypedDict, TypeVar

T = TypeVar("T")
Timed = Tuple[float, T]


class AiTree(TypedDict):
    is_solved: bool
    ai_score: float
    expandable: list[Tuple[str, Optional[int]]]
    in_stock: list[Tuple[str, Optional[int]]]
    children: list[Optional["AiTree"]]


class Setup(TypedDict):
    agg: Literal["max", "min"]
    score: Literal["sa", "sc", "mf"]
    uw_multiplier: float
    normalize: Tuple[float, float, bool]


class AiInput(TypedDict):
    smiles: str
    setup: Setup
