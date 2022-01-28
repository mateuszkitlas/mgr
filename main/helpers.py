from contextlib import asynccontextmanager

from .conda_app import CondaApp
from .score import Score, Smiles
from .types import AiTree, RawScore, Timed


def app_ai():
    return CondaApp[str, Timed[AiTree]](4000, "ai", "aizynth-env")


class _AppScorers:
    def __init__(self):
        self.all: list[Score[float]] = []
        self.app = CondaApp[str, RawScore](4001, "scorers", "scorers")

    async def score(self, smiles: str) -> Smiles:
        raw_score = await self.app.fetch(smiles)
        times, score = Score.from_raw(raw_score)
        self.all.append(times)
        return Smiles(smiles, score)

    def print_stats(self, real_time: float):
        print(
            {
                "smiles_count": len(self.all),
                "total_time": sum((s.sa + s.sc + s.ra + s.syba for s in self.all)),
                "real_time": real_time,
            }
        )


@asynccontextmanager
async def app_scorers():
    x = _AppScorers()
    async with x.app:
        yield x
