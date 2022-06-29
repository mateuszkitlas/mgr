from main.data import data

from shared import Db
from .types import AiInput, Setup


uw_multipliers = [0.15, 0.4, 0.6]


def n(first: float, second: float):
    return (first, second, False)

zero_setup: Setup = {
    "score": "sc",
    "uw_multiplier": 0.0,
    "normalize": n(2.5, 4.5),
    "agg": "max",
}

_setups: list[Setup] = [
    zero_setup,
    *[
        {
            "score": "sc",
            "uw_multiplier": uw_multiplier,
            "normalize": normalize,
            "agg": "max",
        }
        for uw_multiplier in uw_multipliers
        for normalize in [n(2.5, 4.5), n(3.0, 4.5), n(3.5, 4.5)]
    ],
    *[
        {
            "score": "sa",
            "uw_multiplier": uw_multiplier,
            "normalize": normalize,
            "agg": "max",
        }
        for uw_multiplier in uw_multipliers
        for normalize in [n(3.0, 6.0), n(4.0, 6.0), n(5.0, 6.0)]
    ],
    *[
        {
            "score": "mf",
            "uw_multiplier": uw_multiplier,
            "normalize": (-700.0, 0.0, True),
            "agg": "min",
        }
        for uw_multiplier in uw_multipliers
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
                for setup_no, setup in enumerate(_setups):
                    print(f"[ai][setup {setup_no}/{len(_setups)}][mol {mol_no}/{count}]")
                    ai_input: AiInput = {"smiles": mol.smiles, "setup": setup}
                    yield db, ai_input, mol
