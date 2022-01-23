from .types import Scores, Tree
from main.conda_app import CondaApp
from .utils import Saver, data
from asyncio import run, gather
import logging

logger = logging.Logger(__name__)
logging.basicConfig(level=logging.DEBUG)

async def main():
    async with CondaApp[str, Tree](4000, "ai", "aizynth-env") as ai, CondaApp[
        str, Scores
    ](4001, "scorers", "scorers") as scorers:
        mols = data(True)
        # saver = Saver(f"{current_dir()}/out.json")
        for i, mol in enumerate(mols):
            print(f"{i}/{len(mols)}")
            tree, score = await gather(ai(mol.smiles), scorers(mol.smiles))
            # print(tree)
            print(score)
            # saver.append(tree.json())


if __name__ == "__main__":
    try:
        run(main())
    except KeyboardInterrupt:
        pass
