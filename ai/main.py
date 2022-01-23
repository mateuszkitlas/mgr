from typing import Any, Optional
from shared import serve, project_dir
import os


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
        return self.finder.tree.serialize2()


if __name__ == "__main__":
    ai_zynth = AiZynth()

    def handler(data: Optional[str]):
        return data and ai_zynth.tree(data)

    serve(handler)
