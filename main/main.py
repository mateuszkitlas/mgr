import sys
from asyncio import run
from typing import Tuple

from shared import Timer

from .data import data, load_trees, save_trees
from .helpers import app_ai, app_scorers, zero_setup
from .tree import Tree
from .types import AiInput, Setup

uw_multipliers = [0.15, 0.4, 0.6]


def n(first: float, second: float):
    return (first, second, False)


setups: list[Setup] = [
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


def _hash(ai_input: AiInput):
    s = ai_input["setup"]
    n = s["normalize"]
    return (
        ai_input["smiles"]
        + s["score"]
        + str(s["uw_multiplier"])
        + str(n[0])
        + str(n[1])
        + str(n[2])
        + s["agg"]
    )


json_file = "trees_v3.json"


async def main():
    mols = data()
    trees = load_trees(json_file)
    done = {_hash(ai_input) for ai_input, _ in trees}
    async with app_ai() as ai:  # , app_scorers() as scorer:
        for j, mol in enumerate(mols):
            for i, setup in enumerate(setups):
                ai_input: AiInput = {"smiles": mol.smiles, "setup": setup}
                txt = f"[setup {i}/{len(setups)}][mol {j}/{len(mols)}]"
                if _hash(ai_input) in done:
                    print(f"{txt}[skipped]")
                else:
                    print(f"{txt} ", end="", flush=True)
                    ai_tree = await ai.tree(setup, mol.smiles)
                    # print(f"][from_ai...", end = "", flush=True)
                    # real_time, tree = await Timer.acalc(
                    #     Tree.from_ai(ai_tree, scorer.score)
                    # )
                    # scorer.add_real_time(real_time)
                    # trees.append((ai_input, tree.json()))
                    trees.append((ai_input, ai_tree))
                    save_trees(trees, json_file)


def dummy_scorings():
    trees = load_trees(json_file)
    per_mol: dict[str, list[Tuple[Setup, Tree]]] = {
        ai_input["smiles"]: [] for ai_input, _ in trees
    }
    for ai_input, ai_tree in trees:
        tree = Tree.from_ai_dummy_scorings(ai_tree)
        per_mol[ai_input["smiles"]].append((ai_input["setup"], tree))
    for smiles, lst in per_mol.items():
        print(smiles)
        for setup, tree in lst:
            stats = tree.stats()
            prefix = (
                "zero_setup"
                if setup["uw_multiplier"] == 0.0
                else f"{setup['score']}, n{setup['normalize'][0]}, uw{setup['uw_multiplier']}"
            )
            print(
                f"{prefix} ",
                {
                    "internal": stats["internal"]["count"],
                    "solved": stats["solved"]["count"],
                    "not_solved": sum(
                        [s["count"] for s in stats["not_solved"].values()]
                    ),
                },
            )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "dummy_scorings":
        dummy_scorings()
    elif len(sys.argv) == 1:
        run(main())
    else:
        print("invalid sys.argv")
