import asyncio
from typing import Optional, Tuple
from unittest import IsolatedAsyncioTestCase, main

from shared import Fn, Timer

from .data import read_csv
from .helpers import app_ai, app_scorers
from .score import Score, Smiles
from .tree import Tree


class Test(IsolatedAsyncioTestCase):
    async def test_ai(self):
        async with app_ai() as ai:
            ai_tree = await ai(zero_setup, paracetamol.smiles)

            async def fake_scorer(x: Tuple[str, Optional[int]]):
                smiles, transforms = x
                return Smiles(smiles, Score(0.0, 0.0, 0.0, 0.0, 0.0), transforms)

            await Tree.from_ai(ai_tree, fake_scorer)

    async def _test_scorers(self, mols: list[Smiles], test_fn: Fn[Score, bool]):
        async with app_scorers() as scorer:
            real_time, smiles = await Timer.acalc(
                asyncio.gather(*(scorer.score((m.smiles, m.transforms)) for m in mols))
            )
            scorer.add_real_time(real_time)
            failed: list[str] = []
            for test_mol, smiles in zip(mols, smiles):
                diff = Score(
                    sa=abs(test_mol.score.sa - smiles.score.sa),
                    sc=abs(test_mol.score.sc - smiles.score.sc),
                    ra=abs(test_mol.score.ra - smiles.score.ra),
                    mf=abs(test_mol.score.mf - smiles.score.mf),
                    syba=abs(test_mol.score.syba - smiles.score.syba),
                )
                if test_fn(diff):
                    failed.append(
                        "\n" + str(Smiles(smiles.smiles, diff, smiles.transforms))
                    )
            if failed:
                self.fail("".join(failed))

    async def test_mf(self):
        if disable_mf():
            self.skipTest("DISABLE_MF")

        def test_fn(diff: Score):
            return diff.mf > 0.0001

        aspirin = Smiles(
            "O=C(C)Oc1ccccc1C(=O)O",
            Score(0.0, 0.0, 0.0, 10.232122084989555, 0.0),
            None,
        )
        cholesterol = Smiles(
            "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C",
            Score(0.0, 0.0, 0.0, -412.95839636, 0.0),
            None,
        )

        await self._test_scorers([aspirin, cholesterol], test_fn)

    async def test_scorers(self):
        _disable_syba = disable_syba()
        with read_csv(f"test.csv", newline="\r\n", delimiter="\t") as reader:
            # header: SMILES  SAscore SCScore SYBA    RAscore
            mols = [
                Smiles(
                    row[0],
                    Score(
                        sa=float(row[1]),
                        sc=float(row[2]),
                        ra=float(row[4]),
                        mf=0.0,
                        syba=float(row[3]),
                    ),
                    None,
                )
                for row in reader
            ]

        def test_fn(diff: Score):
            return (
                diff.sa > 0.0001
                or diff.sc > 0.0001
                or diff.ra > 0.0001
                or (False if _disable_syba else diff.syba > 0.0001)
            )

        await self._test_scorers(mols, test_fn)

    async def test_all(self):
        async with app_ai() as ai, app_scorers() as scorer:
            ai_tree = await ai.tree(zero_setup, paracetamol.smiles)
            real_time, tree = await Timer.acalc(Tree.from_ai(ai_tree, scorer.score))
            self.assertEqual(tree.not_solved_depth, -1)
            scorer.add_real_time(real_time)
            save_trees([(paracetamol.smiles, tree.json())], "test_all.json")
            loaded_tree = Tree(load_trees("test_all.json")[0][1])
            self.assertEqual(
                [n.ai_score for n in tree.all_nodes()],
                [n.ai_score for n in loaded_tree.all_nodes()],
            )


if __name__ == "__main__":
    main()
