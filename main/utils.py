from typing import Iterable, TypeVar

T = TypeVar("T")


def flatten(l: Iterable[Iterable[T]]) -> Iterable[T]:
    return (e for sl in l for e in sl)
