from shared import Timer
from .score import Score
from .data import test_data, paracetamol
from .helpers import app_ai, app_scorers
import logging
from unittest import IsolatedAsyncioTestCase, main
import asyncio

logger = logging.Logger(__name__)
logging.basicConfig(level=logging.DEBUG)


class Test(IsolatedAsyncioTestCase):
    async def test_ai(self):
        try:
            async with app_ai as ai:
                time, _ai_tree = await ai(paracetamol)
                print(f"Aizync paracetamol time: {time}")
        except KeyboardInterrupt:
            pass

    async def test_scorers(self):
        test_mols = test_data()
        total_time_per_score = Score[float](0.0, 0.0, 0.0, 0.0)
        async with app_scorers as scorers:
            real_timer = Timer()
            raw_scores = await asyncio.gather(*(scorers(m.smiles) for m in test_mols))
            for test_mol, raw_score in zip(test_mols, raw_scores):
                # raw_score = await scorers(test_mol.smiles)
                timed_score = Score.from_raw(raw_score)
                total_time_per_score = timed_score.map_with(
                    float, total_time_per_score, lambda a, b: a[0] + b
                )
                test_mol.score().map_with(
                    float, timed_score, lambda expected, raw: abs(raw[1] - expected),
                ).map(lambda v: self.assertLess(v, 0.0001, test_mol.smiles))
            real_timer.done()
        print("Avarage timings", total_time_per_score.map(lambda v: v / len(test_mols)))
        total_time = (
            total_time_per_score.sa
            + total_time_per_score.sc
            + total_time_per_score.ra
            + total_time_per_score.syba
        )
        print(f"Total time: {total_time}, Real time: {real_timer.delta}")


if __name__ == "__main__":
    main()
