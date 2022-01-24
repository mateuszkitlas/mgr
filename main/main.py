from .types import RawScore, AiTree, Timed
from .score import Score
from .conda_app import CondaApp
from .data import Saver, data
from asyncio import run, gather
import logging

logger = logging.Logger(__name__)
logging.basicConfig(level=logging.DEBUG)


async def main():
    mols = data(True)
    app_ai = CondaApp[str, Timed[AiTree]](4000, "ai", "aizynth-env")
    app_scorers = CondaApp[str, RawScore](4001, "scorers", "scorers")
    async with app_ai as ai, app_scorers as scorers:
        for i, test_mol in enumerate(mols):
            print(f"{i}/{len(mols)}")
            tree, raw_score = await gather(
                ai(test_mol.smiles), scorers(test_mol.smiles)
            )
            timed_score = Score.from_raw(raw_score)
            print(timed_score)


if __name__ == "__main__":
    run(main())
