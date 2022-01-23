from main.conda_app import CondaApp
from main.types import Scores
from .utils import test_data
import logging
from unittest import IsolatedAsyncioTestCase, main

logger = logging.Logger(__name__)
logging.basicConfig(level=logging.DEBUG)


class Test(IsolatedAsyncioTestCase):
    async def test(self):
        try:
            async with CondaApp[str, Scores](4000, "scorers", "scorers") as scorers:
                mols = test_data()
                for i, mol in enumerate(mols):
                    print(f"{i}/{len(mols)}")
                    score = await scorers(mol.smiles)
                    delta: Scores = {
                        "sc": mol.sc - score["sc"],
                        "sa": mol.sa - score["sa"],
                        "ra": mol.ra - score["ra"],
                        "syba": mol.syba - score["syba"],
                    }
                    self.assertLess(sum(delta.values()), 0.0001, mol.smiles)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
