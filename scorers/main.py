import os
import sys
from typing import Callable

from shared import DISABLE_SYBA, Timer, project_dir, serve

# from .scscore_tensorflow import get_sc_scorer


def get_sc_scorer() -> Callable[[str], float]:
    from .scscore_numpy import SCScorer

    scscorer = SCScorer()
    scscorer.restore()

    def sc_score(smiles: str) -> float:
        _, score = scscorer.get_score_from_smi(smiles)
        return score

    return sc_score


def get_syba_scorer() -> Callable[[str], float]:
    from syba.syba import SybaClassifier

    syba = SybaClassifier()
    syba.fitDefaultScore()

    def scorer(smiles: str) -> float:
        return syba.predict(smiles)

    return scorer


def get_ra_scorer(
    model: str,  # Literal["DNN", "XGB"],
    db: str,  # Literal["chembl", "gdbchembl", "gdbmedchem"]
) -> Callable[[str], float]:
    dnn = model == "DNN"
    if dnn:
        from RAscore import RAscore_NN

        f = RAscore_NN.RAScorerNN
    else:
        from RAscore import RAscore_XGB

        f = RAscore_XGB.RAScorerXGB
    x = "fcfp" if dnn else "ecfp"
    y = "h5" if dnn else "pkl"
    pth = f"{project_dir}/data/ra_models/{model}_{db}_{x}_counts/model.{y}"
    _model = f(pth)

    def scorer(smiles: str) -> float:
        return _model.predict(smiles).item()

    return scorer


def get_sa_scorer() -> Callable[[str], float]:
    from rdkit import Chem
    from rdkit.Chem import RDConfig

    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer

    def scorer(smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        return sascorer.calculateScore(mol)

    return scorer


if __name__ == "__main__":

    ra_time, ra_scorer = Timer.calc(lambda: get_ra_scorer("DNN", "chembl"))
    sa_time, sa_scorer = Timer.calc(get_sa_scorer)
    sc_time, sc_scorer = Timer.calc(get_sc_scorer)
    syba_time: float
    syba_scorer: Callable[[str], float]
    if DISABLE_SYBA:
        print("DISABLE_SYBA")

        def f(_: str) -> float:
            return 0.0

        syba_time, syba_scorer = 0.0, f
    else:
        print("Loading syba scorer. It's gonna take ~2 minutes.")
        syba_time, syba_scorer = Timer.calc(get_syba_scorer)
        times = {"ra": ra_time, "sa": sa_time, "sc": sc_time, "syba": syba_time}
        print(f"Syba loaded. Loading times: {times}")

    def scorer(smiles: str):
        return smiles and {
            "ra": Timer.calc(lambda: ra_scorer(smiles)),
            "sa": Timer.calc(lambda: sa_scorer(smiles)),
            "sc": Timer.calc(lambda: sc_scorer(smiles)),
            "syba": Timer.calc(lambda: syba_scorer(smiles)),
        }

    scorer("CC(=O)Nc1ccc(O)cc1")  # warm up

    serve(scorer)
