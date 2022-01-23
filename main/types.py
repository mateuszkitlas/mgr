from typing import TypedDict


class Scores(TypedDict):
    sa: float
    sc: float
    ra: float
    syba: float


class Tree(TypedDict):
    is_solved: bool
    score: float
    expandable_smileses: list[str]
    children: list["Tree"]
