import logging
from asyncio import gather, run

from .data import Saver, data
from .helpers import app_ai, app_scorers

logger = logging.Logger(__name__)
logging.basicConfig(level=logging.DEBUG)


async def main():
    mols = data(True)
    async with app_ai() as ai, app_scorers() as scorer:
        for i, test_mol in enumerate(mols):
            print(f"{i}/{len(mols)}")
            tree, smiles = await gather(
                ai.tree(test_mol.smiles), scorer.score(test_mol.smiles)
            )


if __name__ == "__main__":
    run(main())
