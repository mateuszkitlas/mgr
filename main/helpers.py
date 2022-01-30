from contextlib import asynccontextmanager
from typing import Any, Optional, Tuple

from main.utils import serialize_dict

from .conda_app import CondaApp
from .score import JsonScore, Score, Smiles
from .types import AiTree, Timed


def _print_stats(j: dict[str, Any]):
    print(f"{serialize_dict(j, ', ')}")


class _AppScorers:
    def __init__(self):
        self.all: list[Score] = []
        self.conda_app = CondaApp[str, Tuple[JsonScore, JsonScore]](
            4001, "scorers", "scorers"
        )
        self.real_time = 0.0

    async def score(self, x: Tuple[str, Optional[int]]) -> Smiles:
        smiles, transforms = x
        jtimes, jscore = await self.conda_app.fetch(smiles)
        times, score = Score.from_json(jtimes), Score.from_json(jscore)
        self.all.append(times)
        return Smiles(smiles, score, transforms)

    def add_real_time(self, real_time: float):
        self.real_time += real_time

    def stats(self):
        return {
            "app": str(self.conda_app),
            "smiles_count": len(self.all),
            "total_time": sum((s.sa + s.sc + s.ra + s.syba for s in self.all)),
            "real_time": self.real_time,
        }


@asynccontextmanager
async def app_scorers():
    app = _AppScorers()
    try:
        async with app.conda_app:
            yield app
    finally:
        _print_stats(app.stats())


class _AppAi:
    def __init__(self):
        self.all: list[float] = []
        self.conda_app = CondaApp[str, Timed[AiTree]](4000, "ai", "aizynth-env")

    async def tree(self, smiles: str) -> AiTree:
        times, ai_tree = await self.conda_app.fetch(smiles)
        self.all.append(times)
        return ai_tree

    def stats(self):
        return {
            "app": str(self.conda_app),
            "smiles_count": len(self.all),
            "total_time": sum(self.all),
        }


@asynccontextmanager
async def app_ai():
    app = _AppAi()
    try:
        async with app.conda_app:
            yield app
    finally:
        _print_stats(app.stats())
