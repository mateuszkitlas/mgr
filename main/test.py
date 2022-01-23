from main.conda_app import CondaApp
from main.types import RaSaScScore
from .utils import test_data
import logging
from unittest import IsolatedAsyncioTestCase, main

logger = logging.Logger(__name__)
logging.basicConfig(level=logging.DEBUG)


class Test(IsolatedAsyncioTestCase):
    async def test(self):
        try:
            async with CondaApp[str, RaSaScScore](4000, "ra-sa", "ra-sa") as ra_sa_sc:
                mols = test_data()
                for i, mol in enumerate(mols):
                    print(f"{i}/{len(mols)}")
                    score = await ra_sa_sc(mol.smiles)
                    delta: RaSaScScore = {
                        "sc": mol.sc - score["sc"],
                        "sa": mol.sa - score["sa"],
                        "ra": mol.ra - score["ra"],
                    }
                    self.assertLess(sum(delta.values()), 0.0001, mol.smiles)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
