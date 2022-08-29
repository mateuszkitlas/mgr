from main.data import data
from shared import Db

from .types import AiInput, Setup

other_score_multipliers = [0.2375, 0.475, 0.7125]  # 0.95/4 * 1, *2, *3,
aggs = ["min", "max"]


def n(first: float, second: float):
    return (first, second, False)


zero_setup: Setup = {
    "score": "sc",
    "uw_multiplier": 0.0,
    "agg": "max",
}

ai_setups: list[Setup] = [
    zero_setup,
    *[
        {
            "score": "sc",
            "uw_multiplier": mul,
            "agg": agg,
        }
        for mul in other_score_multipliers
        for agg in aggs
    ],
    *[
        {
            "score": "sa",
            "uw_multiplier": mul,
            "agg": agg,
        }
        for mul in other_score_multipliers
        for agg in aggs
    ],
    *[
        {
            "score": "syba",
            "uw_multiplier": mul,
            "agg": agg,
        }
        for mul in other_score_multipliers
        for agg in aggs
    ],
]


def ai_input_gen(readonly: bool, only_zero_setup: bool):
    with Db("db", readonly) as db:
        mols = data()
        count = len(mols)
        for mol_no, mol in enumerate(mols):
            if only_zero_setup:
                ai_input: AiInput = {"smiles": mol.smiles, "setup": zero_setup}
                yield db, ai_input, mol
            else:
                for setup_no, setup in enumerate(ai_setups):
                    print(
                        f"[ai][setup {setup_no}/{len(ai_setups)}][mol {mol_no}/{count}]"
                    )
                    ai_input: AiInput = {"smiles": mol.smiles, "setup": setup}
                    yield db, ai_input, mol

def ai_mol_gen():
    mols = data()
    for mol_no, mol in enumerate(mols):
        yield mol_no, mol

def ai_setup_gen(readonly: bool, only_zero_setup: bool):
    with Db("db", readonly) as db:
        if only_zero_setup:
            ai_input: AiInput = {"smiles": mol.smiles, "setup": zero_setup}
            yield db, zero_setup
        else:
            for setup_no, setup in enumerate(ai_setups):
                print(
                    f"[ai][setup {setup_no}/{len(ai_setups)}]"
                )
                yield db, setup
