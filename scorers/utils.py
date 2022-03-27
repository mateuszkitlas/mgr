import os
import sys

from shared import Fn, disable_mf, disable_syba, project_dir

# from .scscore_tensorflow import get_sc_scorer

Scorer = Fn[str, float]


def dummy_scorer(_smiles: str):
    return 0.0


def get_mf_scorer() -> Scorer:
    if disable_mf():
        return dummy_scorer
    else:
        from .mf_score.mfscore.ocsvm import score_smiles

        return score_smiles


def get_sc_scorer() -> Scorer:
    from .scscore_numpy import SCScorer

    scscorer = SCScorer()
    scscorer.restore()

    def sc_score(smiles: str) -> float:
        _, score = scscorer.get_score_from_smi(smiles)
        return score

    return sc_score


def get_syba_scorer() -> Scorer:
    if disable_syba():
        return dummy_scorer
    else:
        print("Loading syba scorer. It's gonna take ~2 minutes.")
        from syba.syba import SybaClassifier

        syba = SybaClassifier()
        syba.fitDefaultScore()

        def scorer(smiles: str) -> float:
            return syba.predict(smiles)

        return scorer


def get_ra_scorer(
    model: str,  # Literal["DNN", "XGB"],
    db: str,  # Literal["chembl", "gdbchembl", "gdbmedchem"]
) -> Scorer:
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


def get_sa_scorer() -> Scorer:
    from rdkit import Chem
    from rdkit.Chem import RDConfig

    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer

    def scorer(smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        return sascorer.calculateScore(mol)

    return scorer
