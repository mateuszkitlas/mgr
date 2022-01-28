import os
from argparse import ArgumentParser
from asyncio import run
from typing import Tuple

from .data import data, load_trees, paracetamol, save_trees
from .helpers import app_ai, app_scorers
from .stats import scatter_pairs
from .tree import JsonTree, Tree


async def trees():
    mols = data()
    trees: list[Tuple[str, JsonTree]] = load_trees("trees.json")
    done = {smiles for smiles, _ in trees}
    async with app_ai() as ai, app_scorers() as scorer:
        for i, mol in enumerate(mols):
            print(f"{i}/{len(mols)}")
            if mol.smiles not in done:
                ai_tree = await ai.tree(mol.smiles)
                tree = await Tree.from_ai(ai_tree, scorer.score)
                trees.append((mol.smiles, tree.json()))
                save_trees(trees, "trees.json")


def test_stats():
    [(_, json_tree)] = load_trees("test_all.json")
    scatter_pairs([(paracetamol, Tree(json_tree))])


def stats():
    mol_by_smiles = {mol.smiles: mol for mol in data()}
    mols = [
        (mol_by_smiles[smiles], Tree(json_tree))
        for (smiles, json_tree) in load_trees("trees.json")
    ]
    scatter_pairs(mols)


if __name__ == "__main__":
    a = ArgumentParser()
    a.add_argument(
        "action", default="all", choices=["all", "trees", "stats", "test-stats"]
    )
    a.add_argument("--disable-syba", action="store_true")
    args = a.parse_args()
    if args.disable_syba:
        os.environ["DISABLE_SYBA"] = "1"
    if args.action == "trees":
        run(trees())
    elif args.action == "test-stats":
        test_stats()
    else:
        raise NotImplementedError
