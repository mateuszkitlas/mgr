from contextlib import asynccontextmanager

from .conda_app import CondaApp
from .score import Score, Smiles
from .types import AiTree, RawScore, Timed


class _AppScorers:
    def __init__(self):
        self.all: list[Score[float]] = []
        self.conda_app = CondaApp[str, RawScore](4001, "scorers", "scorers")

    async def score(self, smiles: str) -> Smiles:
        raw_score = await self.conda_app.fetch(smiles)
        times, score = Score.from_raw(raw_score)
        self.all.append(times)
        return Smiles(smiles, score)

    def print_stats(self, real_time: float):
        print(
            {
                "app": str(self.conda_app),
                "smiles_count": len(self.all),
                "total_time": sum((s.sa + s.sc + s.ra + s.syba for s in self.all)),
                "real_time": real_time,
            }
        )


@asynccontextmanager
async def app_scorers():
    app = _AppScorers()
    async with app.conda_app:
        yield app


class _AppAi:
    def __init__(self):
        self.all: list[float] = []
        self.conda_app = CondaApp[str, Timed[AiTree]](4000, "ai", "aizynth-env")

    async def tree(self, smiles: str) -> AiTree:
        times, ai_tree = await self.conda_app.fetch(smiles)
        self.all.append(times)
        return ai_tree

    def print_stats(self):
        print(
            {
                "app": str(self.conda_app),
                "smiles_count": len(self.all),
                "total_time": sum(self.all),
            }
        )


@asynccontextmanager
async def app_ai():
    app = _AppAi()
    async with app.conda_app:
        yield app
