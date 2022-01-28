from typing import Any, Iterable, Type, TypeVar

T = TypeVar("T")


def flatten(l: Iterable[Iterable[T]]) -> Iterable[T]:
    return (e for sl in l for e in sl)


def cast(t: Type[T], v: Any) -> T:
    return v
