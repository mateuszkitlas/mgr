import os
from typing import Any, Optional

from aizynthfinder.aizynthfinder import AiZynthFinder

from shared import Timer, project_dir, serve


def serialize_state(state):
    in_stock_list = [mol in state.stock for mol in state.mols]
    expandable = [
        (mol.smiles, mol.transform) for mol, in_stock in zip(state.mols, in_stock_list) if not in_stock
    ]
    in_stock = [
        (mol.smiles, mol.transform) for mol, in_stock in zip(state.mols, in_stock_list) if in_stock
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
    return os.path.join(project_dir, "data", filename)


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


if __name__ == "__main__":

    def handler(smiles: Optional[str]):
        return smiles and Timer.calc(lambda: AiZynth.calc(smiles))

    serve(handler)
