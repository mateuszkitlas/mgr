import asyncio
import os
from typing import (Any, Callable, Dict, List, Literal, Optional, Tuple,
                    TypedDict)

import numpy as np

from shared import CondaApp, Timer, project_dir, serve


class Setup(TypedDict):
    agg: Literal["max", "min"]
    score: Literal["sa", "sc", "ra", "syba", "mf"]
    uw_multiplier: float
    # normalize: Literal["sa", "sc", "ra", "syba", "mf"]


class Input(TypedDict):
    smiles: str
    setup: Setup


setup: Optional[Setup] = None


def patch_calc_score(scorer: Callable[[List[str], float, float], float]):
    from aizynthfinder.search.mcts.state import MctsState

    def _calc_score(self):
        # How many is in stock (number between 0 and 1)
        num_in_stock = np.sum(self.in_stock_list)
        # This fraction in stock, makes the algorithm artificially add stock compounds by cyclic addition/removal
        fraction_in_stock = num_in_stock / len(self.mols)

        # Max_transforms should be low
        max_transforms = self.max_transforms
        # Squash function, 1 to 0, 0.5 around 4.
        max_transforms_score: float = self._squash_function(max_transforms, -1, 0, 4)

        expandable: List[str] = [
            mol.smiles for mol in self.mols if mol not in self.stock
        ]
        return scorer(expandable, fraction_in_stock, max_transforms_score)

    MctsState._calc_score = _calc_score


def serialize_state(state):
    in_stock_list = [mol in state.stock for mol in state.mols]
    expandable = [
        (mol.smiles, mol.transform)
        for mol, in_stock in zip(state.mols, in_stock_list)
        if not in_stock
    ]
    in_stock = [
        (mol.smiles, mol.transform)
        for mol, in_stock in zip(state.mols, in_stock_list)
        if in_stock
    ]
    return {
        "ai_score": state.score,
        "expandable": expandable,
        "in_stock": in_stock,
    }


def serialize_tree(tree):
    return {
        "children": [
            serialize_tree(child) if child else None for child in tree._children
        ],
        "is_solved": tree.is_solved,
        **serialize_state(tree._state),
    }


def file(filename: str):
    return os.path.join(project_dir, "data/ai", filename)


configdict = {
    "policy": {
        "files": {
            "full_uspto": [
                file("full_uspto_03_05_19_rollout_policy.hdf5"),
                file("full_uspto_03_05_19_unique_templates.hdf5"),
            ],
        },
    },
    "stock": {"files": {"zinc": file("zinc_stock_17_04_20.hdf5")}},
}


class AiZynth:
    def __init__(self):
        from aizynthfinder.aizynthfinder import AiZynthFinder

        self.finder = AiZynthFinder(configdict=configdict)
        self.finder.stock.select("zinc")
        self.finder.expansion_policy.select("full_uspto")
        # self.finder.filter_policy.select("full_uspto")
        # self.expander = AiZynthExpander(self.config_path)
        # self.expander.expansion_policy.select("full_uspto")
        # self.expander.filter_policy.select("uspto")

    def tree(self, smiles: str) -> Any:
        self.finder.target_smiles = smiles
        self.finder.tree_search()
        self.finder.build_routes()
        self.finder.routes.compute_scores(*self.finder.scorers.objects())
        return serialize_tree(self.finder.tree.root)

    @staticmethod
    def calc(smiles: str):
        return AiZynth().tree(smiles)


def score_transformer(score_value: float, score_type: str) -> float:
    """Transform score so that fall within [0,1].

    0 means infeasible (not-accessible) molecule, 1 means fully feasible
    molecule.
    """
    if score_type == "sa":
        # Transform [1, 10] to [0,1] and invert
        result = 1 - (score_value - 1) / 9
    elif score_type == "sc":
        # Transform [1, 5] to [0,1] and invert
        result = 1 - ((score_value - 1) / 4)
    elif score_type == "ra":
        result = score_value
    elif score_type == "syba":
        # For now suppose it is within [-100, 100]
        result = (min(max(score_value, -100), 100) / 200) + 0.5
    elif score_type == "mf":
        # Transform [-800, 600] so that 0 turns into 0.5 (no inversion)
        truncated = min(max(score_value, -800), 600)
        if truncated > 0:
            truncated /= 2 * 600
        else:
            truncated /= 2 * 800
        result = truncated + 0.5
    else:
        result = score_value
    assert 0 <= result <= 1
    return result


def normalize(score_value: float, score_type: str):
    return score_transformer(score_value, score_type)

async def main(port=4002):
    conda_app = CondaApp[Tuple[str, List[str]], List[float]](port, "scorers", "scorers")

    async with conda_app as (_, fetch_sync):
        cache: Dict[Tuple[str, str], float] = {}

        def fetch(scoring: str, smileses: List[str]):
            not_in_cache: list[str] = [
                smiles for smiles in smileses if (scoring, smiles) not in cache
            ]
            for smiles, score in zip(not_in_cache, fetch_sync((scoring, not_in_cache))):
                cache[(scoring, smiles)] = score
            return (cache[(scoring, smiles)] for smiles in smileses)

        def scorer(
            expandable: List[str], fraction_in_stock: float, max_transforms_score: float
        ):
            if setup:
                uw_mul = setup["uw_multiplier"]
                ai_mul = 0.95 - uw_mul
                mt_mul = 0.05

                def f(normalized: float):
                    return (
                        uw_mul * normalized
                        + ai_mul * fraction_in_stock
                        + mt_mul * max_transforms_score
                    )

                if uw_mul > 0.0 and expandable:
                    agg = {"min": min, "max": max}[setup["agg"]]
                    score_type = setup["score"]
                    scores = fetch(setup["score"], expandable)
                    normalized = normalize(agg(scores), score_type)
                    return 0 if normalized is None else f(normalized)
                    # Probably now normalized is never None
                else:
                    return f(0.0)
            else:
                raise NotImplementedError

        patch_calc_score(scorer)

        def handler(data: Optional[Input]):
            if data:
                global setup
                setup = data["setup"]
                return Timer.calc(lambda: AiZynth.calc(data["smiles"]))

        serve(handler)


if __name__ == "__main__":
    asyncio.run(main())
