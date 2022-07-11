import asyncio
from contextlib import asynccontextmanager, contextmanager
from typing import List, Literal, Optional, Tuple

from main.score import Score, Smiles
from main.utils import dict_gather
from shared import CondaApp, Db

from .types import AcResult, AiInput, AiTree, Timed

Scoring = Literal["sa", "sc", "ra", "syba", "mf", "mfgb"]
all_scorings: list[Scoring] = ["sa", "sc", "ra", "syba", "mf", "mfgb"]


@asynccontextmanager
async def app_scorers():
    async with CondaApp[Tuple[str, List[str]], List[float]](
        4002, "scorers", "scorers"
    ) as (fetch, _):

        async def f(scoring: Scoring, smiles: str):
            return (await fetch((scoring, [smiles])))[0]

        async def g(data: Tuple[str, Optional[int]]):
            s, t = data
            v: dict[Scoring, float] = await dict_gather(
                {k: f(k, s) for k in all_scorings}
            )
            return Smiles(s, Score(**v), t)

        yield f, g


@contextmanager
def db_scorers():
    with Db("scores", True) as db:

        def f(scoring: Scoring, smiles: str):
            return db.read(
                [scoring, db.read(["smiles", smiles], str, True)], float, False
            )

        def g(data: Tuple[str, Optional[int]]):
            s, t = data
            return Smiles(s, Score(**{k: f(k, s) for k in all_scorings}), t)

        async def h(data: Tuple[str, Optional[int]]):
            return g(data)

        yield g, h


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
