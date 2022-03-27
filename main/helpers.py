from contextlib import asynccontextmanager
from typing import Any, Optional, Tuple

from conda_app import CondaApp
from main.utils import serialize_dict

from .score import JsonScore, Score, Smiles
from .types import AiInput, AiTree, Setup, Timed


def _print_stats(j: dict[str, Any]):
    print(f"{serialize_dict(j, ', ')}")


_scorers_cache: dict[str, Score] = {}
_used_scorers_cache = 0


class _AppScorers:
    def __init__(self):
        self.all: list[Score] = []
        self.conda_app = CondaApp[str, Tuple[JsonScore, JsonScore]](
            4001, "scorers.main", "scorers"
        )
        self.real_time = 0.0

    async def score(self, x: Tuple[str, Optional[int]]) -> Smiles:
        smiles, transforms = x
        if smiles in _scorers_cache:
            global _used_scorers_cache
            _used_scorers_cache += 1
            return Smiles(smiles, _scorers_cache[smiles], transforms)
        else:
            jtimes, jscore = await self.conda_app.fetch(smiles)
            times, score = Score.from_json(jtimes), Score.from_json(jscore)
            self.all.append(times)
            return Smiles(smiles, score, transforms)

    def add_real_time(self, real_time: float):
        self.real_time += real_time
        _print_stats(self.stats())

    def stats(self):
        return {
            "app": str(self.conda_app),
            "smiles_count": len(self.all),
            "total_time": sum((s.sa + s.sc + s.ra + s.syba for s in self.all)),
            "real_time": self.real_time,
            "used_scorers_cache": _used_scorers_cache,
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
        self.conda_app = CondaApp[AiInput, Timed[AiTree]](
            4000, "ai.main", "aizynth-env",
        )

    async def tree(self, setup: Setup, smiles: str) -> AiTree:
        times, ai_tree = await self.conda_app.fetch({"setup": setup, "smiles": smiles})
        print(times)
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


zero_setup: Setup = {
    "score": "mf",
    "uw_multiplier": 0.0,
    "normalize": (0.0, 0.0, False),
    "agg": "max",
}
