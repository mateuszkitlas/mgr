import asyncio
from contextlib import asynccontextmanager
from typing import List, Literal, Tuple, Union

from shared import CondaApp

from .types import AcResult, AiInput, AiTree, Timed

Scoring = Literal["sa", "sc", "ra", "syba", "mf"]
all_scorings: List[Scoring] = ["sa", "sc", "ra", "syba", "mf"]


@asynccontextmanager
async def app_scorers():
    app = CondaApp[Tuple[Union[Scoring, Literal["smiles"]], str], Union[float, str]](
        4000, "scorers", "scorers"
    )
    try:
        await app.start()

        async def f(scoring: Scoring, smiles: str) -> float:
            ret = await app.fetch((scoring, smiles))
            assert isinstance(ret, float)
            return ret

        async def g(smiles: str) -> str:
            ret = await app.fetch(("smiles", smiles))
            assert isinstance(ret, str)
            return ret

        yield f, g
    finally:
        app.stop()


@asynccontextmanager
async def app_ai_ac():
    ai = CondaApp[AiInput, Timed[AiTree]](4001, "ai", "aizynth-env")
    ac = CondaApp[str, Timed[AcResult]](4002, "ac", "askcos")
    try:
        await asyncio.gather(ai.start(), ac.start())
        yield (ai.fetch, ac.fetch)
    finally:
        try:
            ai.stop()
        finally:
            ac.stop()


@asynccontextmanager
async def app_ac():
    async with CondaApp[str, Timed[AcResult]](4002, "ac", "askcos") as x:
        yield x


@asynccontextmanager
async def app_ai():
    async with CondaApp[AiInput, Timed[AiTree]](4001, "ai", "aizynth-env") as x:
        yield x
