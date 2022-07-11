from typing import Any, Dict, List, Optional, Tuple

import rdkit.Chem as Chem

from shared import CondaApp, Timer, project_dir, serve


def init():
    from os import symlink
    from os.path import exists

    from askcos.prioritization.prioritizer.scscore import SCScorePrecursorPrioritizer
    from askcos.retrosynthetic.mcts.tree_builder import MCTS
    from askcos.retrosynthetic.transformer import RetroTransformer

    data_dir = project_dir + "/askcos-core/askcos/data"
    if not exists(data_dir):
        symlink(project_dir + "/askcos-data", data_dir)


def f(data: Tuple[Setup, str]):
    class UwScore(SCScorePrecursorPrioritizer):
        def get_score_from_smiles(self, smiles: str, noprice: bool = True) -> float:
            if not noprice:
                ppg: float = self.pricer.lookup_smiles(smiles, alreadyCanonical=True)
                if ppg:
                    return ppg / 100.0

            fp = np.array((self.smi_to_fp(smiles)), dtype=np.float32)
            if sum(fp) == 0:
                cur_score = 0.0
            else:
                # Run
                cur_score = self.apply(fp)
            return cur_score

    retro_transformer = RetroTransformer(
        template_set="reaxys",
        template_prioritizer=None,
        precursor_prioritizer="relevanceheuristic",
        fast_filter=None,
        scscorer=UwScore(),
    )
    retro_transformer.load()
    self.mcts = MCTS(
        nproc=1,
        retroTransformer=retro_transformer,
    )

    def calc(self, raw_smiles: str) -> Tuple[List[Any], Any]:
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(raw_smiles), True)
        paths, _, graph = self.mcts.get_buyable_paths(
            smiles,
            nproc=self.nproc,
            expansion_time=30,
            max_cum_template_prob=0.995,
            template_count=100,
            termination_logic={"or": ["buyable"]},
            soft_reset=True,  # do not kill workers since there are more tests
            soft_stop=True,  # do not kill workers since there are more tests
        )
        return smiles, paths, graph


async def main():
    conda_app = CondaApp[Tuple[str, List[str]], List[float]](4003, "scorers", "scorers")
    async with conda_app as (_, fetch_sync):
        cache: Dict[Tuple[str, str], float] = {}

        def fetch(scoring: str, smileses: List[str]):
            not_in_cache: list[str] = [
                smiles for smiles in smileses if (scoring, smiles) not in cache
            ]
            for smiles, score in zip(not_in_cache, fetch_sync((scoring, not_in_cache))):
                cache[(scoring, smiles)] = score
            return (cache[(scoring, smiles)] for smiles in smileses)


if __name__ == "__main__":
    ac = AskCos()

    def handler(smiles: Optional[str]):
        return smiles and Timer.calc(lambda: ac.calc(smiles))

    serve(handler)
