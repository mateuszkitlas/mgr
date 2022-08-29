import asyncio
from contextlib import asynccontextmanager
from typing import List, Literal, Optional, Tuple

from main.score import Score, Smiles
from shared import CondaApp

from .types import AcResult, AiInput, AiTree, Timed

Scoring = Literal["sa", "sc", "ra", "syba", "mf"]
all_scorings: list[Scoring] = ["sa", "sc", "ra", "syba", "mf"]


@asynccontextmanager
async def app_scorers():
    async with CondaApp[Tuple[str, List[str]], List[float]](
        4002, "scorers", "scorers"
    ) as (fetch, _):

        async def f(scoring: Scoring, smiles: str):
            return (await fetch((scoring, [smiles])))[0]

        async def g(data: Tuple[str, Optional[int]]):
            s, t = data
            sa, sc, ra, syba, mf = await asyncio.gather(
                f("sa", s), f("sc", s), f("ra", s), f("syba", s), f("mf", s)
            )
            return Smiles(s, Score(sa=sa, sc=sc, ra=ra, mf=mf, syba=syba), t)

        yield f, g


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
async def app_ai(offset: float=0):
    async with CondaApp[AiInput, Timed[AiTree]](4001 + offset, "ai", "aizynth-env") as x:
        yield x
