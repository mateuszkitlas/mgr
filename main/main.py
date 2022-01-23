
from typing import TypedDict
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


class X(TypedDict):
    sa: float
    sc: float
    ra: float


async def main():
    async with CondaApp.many(4000, apps) as [ai, ra]:
        mols = data(True)
        #saver = Saver(f"{current_dir()}/out.json")
        for i, mol in enumerate(mols):
            print(f"{i}/{len(mols)}")
            tree, score = await gather(ai(mol.smiles), ra(mol.smiles))
            # print(tree)
            print(score)
            # saver.append(tree.json())

if __name__ == '__main__':
    try:
        run(main())
    except KeyboardInterrupt:
        pass
