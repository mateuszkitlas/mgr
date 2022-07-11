import os
import sys
from typing import List, Optional, Tuple

from shared import Db, Fn, paracetamol_smiles, project_dir, serve

Scorer = Fn[str, float]


def get_mf_scorer() -> Scorer:
    from .mf_score.mfscore.ocsvm import score_smiles

    return score_smiles


def get_mfgb_scorer() -> Scorer:
    from .mf_score.mfscore.gb import score_smiles

    return lambda smiles: score_smiles(smiles)[0].item()


def get_sc_scorer() -> Scorer:
    from .scscore_numpy import SCScorer

    scscorer = SCScorer()
    scscorer.restore()

    return lambda smiles: scscorer.get_score_from_smi(smiles)[1]


def get_syba_scorer() -> Scorer:
    print("Loading syba scorer. It's gonna take ~2 minutes.")
    from syba.syba import SybaClassifier

    syba = SybaClassifier()
    syba.fitDefaultScore()

    return lambda smiles: syba.predict(smiles)


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

    return lambda smiles: _model.predict(smiles).item()


def get_sa_scorer() -> Scorer:
    from rdkit import Chem
    from rdkit.Chem import RDConfig

    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer

    return lambda smiles: sascorer.calculateScore(Chem.MolFromSmiles(smiles))


def get_smileser() -> Fn[str, str]:
    from rdkit import Chem

    return lambda smiles: Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


if __name__ == "__main__":
    try:
        scorers = {
            "ra": get_ra_scorer("DNN", "chembl"),
            "sa": get_sa_scorer(),
            "sc": get_sc_scorer(),
            "mf": get_mf_scorer(),
            "mfgb": get_mfgb_scorer(),
            "syba": get_syba_scorer(),
        }
        smileser = get_smileser()
        with Db("scores", False) as db:

            def get(type: str, smiles: str):
                smiles_canon = db.read_or_create_sync(
                    ["smiles", smiles], lambda: smileser(smiles)
                )
                return db.read_or_create_sync(
                    [type, smiles_canon], lambda: scorers[type](smiles_canon)
                )

            def scorer(data: Optional[Tuple[str, List[str]]]):
                if data:
                    type, smileses = data
                    return [get(type, smiles) for smiles in smileses]

            for key in scorers.keys():
                scorer((key, [paracetamol_smiles]))  # preprocess
                scorer((key, [paracetamol_smiles]))  # read

            serve(scorer)
    except KeyboardInterrupt:
        pass
