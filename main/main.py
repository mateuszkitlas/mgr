from .types import RaSaScScore, Tree
from main.conda_app import CondaApp
from .utils import Saver, data
from asyncio import run, gather
import logging

logger = logging.Logger(__name__)
logging.basicConfig(level=logging.DEBUG)
apps = [
    ("ai", "aizynth-env"),
    ("ra-sa", "ra-sa"),
]


async def main():
    async with CondaApp[str, Tree](4000, "ai", "aizynth-env") as ai, CondaApp[
        str, RaSaScScore
    ](4001, "ra-sa", "ra-sa") as ra_sa_sc:
        mols = data(True)
        # saver = Saver(f"{current_dir()}/out.json")
        for i, mol in enumerate(mols):
            print(f"{i}/{len(mols)}")
            tree, score = await gather(ai(mol.smiles), ra_sa_sc(mol.smiles))
            # print(tree)
            print(score)
            # saver.append(tree.json())


if __name__ == "__main__":
    try:
        run(main())
    except KeyboardInterrupt:
        pass
