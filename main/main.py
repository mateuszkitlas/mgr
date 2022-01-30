from asyncio import run
from typing import Tuple

from shared import Timer

from .data import data, load_trees, save_trees
from .helpers import app_ai, app_scorers
from .tree import JsonTree, Tree


async def main():
    mols = data()
    trees: list[Tuple[str, JsonTree]] = load_trees("trees.json")
    done = {smiles for smiles, _ in trees}
    async with app_ai() as ai, app_scorers() as scorer:
        for i, mol in enumerate(mols):
            print(f"{i}/{len(mols)}")
            if mol.smiles not in done:
                ai_tree = await ai.tree(mol.smiles)
                real_time, tree = await Timer.acalc(Tree.from_ai(ai_tree, scorer.score))
                scorer.add_real_time(real_time)
                trees.append((mol.smiles, tree.json()))
                save_trees(trees, "trees.json")


if __name__ == "__main__":
    run(main())
