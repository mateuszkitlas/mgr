from .conda_app import CondaApp
from .types import RawScore
from .score import Score
from .data import test_data
import logging
from unittest import IsolatedAsyncioTestCase, main

logger = logging.Logger(__name__)
logging.basicConfig(level=logging.DEBUG)

class Test(IsolatedAsyncioTestCase):
    async def test(self):
        test_mols = test_data()
        sum_score = Score[float](0., 0., 0., 0.)
        app_scorers = CondaApp[str, RawScore](4001, "scorers", "scorers")
        n = 0
        try:
            async with app_scorers as scorers:
                for i, test_mol in enumerate(test_mols):
                    print(f"{i}/{len(test_mols)}")
                    raw_score = await scorers(test_mol.smiles)
                    timed_score = Score.from_raw(raw_score)
                    sum_score = timed_score.map_with(float, sum_score, lambda a, b: a[0] + b)
                    test_mol.score().map_with(
                        float,
                        timed_score,
                        lambda expected, raw: abs(raw[1] - expected),
                    ).map(lambda v: self.assertLess(v, 0.0001, test_mol.smiles))
                    n += 1
        except KeyboardInterrupt:
            pass
        print("Avarage timings", sum_score.map(lambda v: v/n))


if __name__ == "__main__":
    main()
