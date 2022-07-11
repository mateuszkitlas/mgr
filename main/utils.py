from asyncio import gather
from typing import Awaitable, Iterable, Optional, TypeVar

_T = TypeVar("_T")
_K = TypeVar("_K")


def not_none(l: Iterable[Optional[_T]]) -> Iterable[_T]:
    return (e for e in l if e is not None)


def flatten(l: Iterable[Iterable[_T]]) -> Iterable[_T]:
    return (e for sl in l for e in sl)


async def dict_gather(d: dict[_K, Awaitable[_T]]) -> dict[_K, _T]:
    keys = list(d.keys())
    return {k: v for k, v in zip(keys, await gather(*[d[k] for k in keys]))}
