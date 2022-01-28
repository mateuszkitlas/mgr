import asyncio
from unittest import IsolatedAsyncioTestCase, main

from shared import Timer

from .data import load_trees, paracetamol, save_trees, test_data
from .helpers import app_ai, app_scorers
from .score import Score, Smiles
from .tree import Tree


class Test(IsolatedAsyncioTestCase):
    async def test_ai(self):
        async with app_ai() as ai:
            ai_tree = await ai.tree(paracetamol.smiles)
            ai.print_stats()

            async def fake_scorer(smiles: str):
                return Smiles(smiles, Score[float](0.0, 0.0, 0.0, 0.0))

            await Tree.from_ai(ai_tree, fake_scorer)

    async def test_scorers(self):
        test_mols = test_data()
        async with app_scorers() as scorer:
            real_time, smiles = await Timer.acalc(
                asyncio.gather(*(scorer.score(m.smiles) for m in test_mols))
            )
            scorer.print_stats(real_time)
            failed: list[str] = []
            for test_mol, smiles in zip(test_mols, smiles):
                diff = test_mol.score.add(
                    smiles.score, lambda expected, actual: abs(actual - expected)
                )
                if (
                    diff.sa > 0.0001
                    or diff.sc > 0.0001
                    or diff.ra > 0.0001
                    or diff.syba > 0.0001
                ):
                    failed.append(str(Smiles(smiles.smiles, diff)))
            if failed:
                self.fail("\n".join(failed))

    async def test_all(self):
        async with app_ai() as ai, app_scorers() as scorer:
            ai_tree = await ai.tree(paracetamol.smiles)
            real_time, tree = await Timer.acalc(Tree.from_ai(ai_tree, scorer.score))
            save_trees([(paracetamol.smiles, tree.json())], "test_all.json")
            loaded_tree = Tree(load_trees("test_all.json")[0][1])
            self.assertEqual(
                [n.ai_score for n in tree.all_nodes()],
                [n.ai_score for n in loaded_tree.all_nodes()],
            )


if __name__ == "__main__":
    main()
