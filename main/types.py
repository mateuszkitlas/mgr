from typing import TypedDict


class RaSaScScore(TypedDict):
    sa: float
    sc: float
    ra: float


class Tree(TypedDict):
    is_solved: bool
    score: float
    expandable_smileses: list[str]
    children: list["Tree"]
