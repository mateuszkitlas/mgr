from typing import Any, Optional
from shared import serve, project_dir, Timer
import os


def serialize_state(state):
    in_stock_list = [mol in state.stock for mol in state.mols]
    expandable_smiles = [
        mol.smiles for mol, in_stock in zip(state.mols, in_stock_list) if not in_stock
    ]
    in_stock_smiles = [
        mol.smiles for mol, in_stock in zip(state.mols, in_stock_list) if in_stock
    ]
    return {
        "score": state.score,
        "expandable_smiles": expandable_smiles,
        "in_stock_smiles": in_stock_smiles,
    }


def serialize_tree(tree):
    return {
        "children": [
            serialize_tree(child) if child else None for child in tree._children
        ],
        "is_solved": tree.is_solved,
        **serialize_state(tree._state),
    }


class AiZynth:
    def __init__(self):
        from aizynthfinder.aizynthfinder import AiZynthFinder

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


if __name__ == "__main__":
    ai_zynth = AiZynth()

    def handler(smiles: Optional[str]):
        return smiles and Timer.calc(lambda: ai_zynth.tree(smiles))

    serve(handler)
