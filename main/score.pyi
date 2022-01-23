from typing import Callable, Generic, Type, TypeVar, Dict
from .types import RawScore, Timed


T = TypeVar("T")
U = TypeVar("U")
R = TypeVar("R")

class Score(Generic[T]):
    sa: T
    sc: T
    ra: T
    syba: T
    def __init__(self, sa: T, sc: T, ra: T, syba: T) -> None: ...
    def map_with(self, _: Type[R], score: Score[U], fn: Callable[[T, U], R]) -> Score[R]: ...
    def map(self, fn: Callable[[T], R]) -> Score[R]: ...
    def __str__(self) -> str: ...
    def json(self) -> Dict[str, T]: ...
    @staticmethod
    def from_raw(raw: RawScore) -> Score[Timed[float]]: ...