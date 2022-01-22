
from main.conda_app import CondaApp
from shared import project_dir
from .utils import Saver, read_csv
from asyncio import run, gather
import logging

logger = logging.Logger(__name__)
logging.basicConfig(level=logging.DEBUG)
apps = [
    ("ai", "aizynth-env"),
    ("ra", "py3.7"),
]


async def main():
    async with CondaApp.many(4000, apps) as [ai, ra]:
        try:
            mols = read_csv(f"{project_dir}/main/data.csv", True)
            #saver = Saver(f"{current_dir()}/out.json")
            for i, mol in enumerate(mols):
                print(f"{i}/{len(mols)}")
                tree, score = await gather(ai(mol.smiles), ra(mol.smiles))
                print(tree)
                print(score)
                # saver.append(tree.json())
        except KeyboardInterrupt:
            pass

if __name__ == '__main__':
    run(main())
