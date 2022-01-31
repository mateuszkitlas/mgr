from typing import Any, Iterable, TypedDict, TypeVar, Union

T = TypeVar("T")


def flatten(l: Iterable[Iterable[T]]) -> Iterable[T]:
    return (e for sl in l for e in sl)


def serialize_dict(d: Union[dict[str, Any], TypedDict], sep: str):
    return sep.join([f"{k}: {v}" for k, v in d.items()])


def human_json(d: Union[dict[str, Any], TypedDict]):
    return "{" + serialize_dict(d, ", ") + "}"
