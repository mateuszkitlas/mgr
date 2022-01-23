from typing import Callable
from shared import serve, project_dir
import sys
import os
from .scscore import get_sc_scorer


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

    def scorer(smiles: str) -> float:
        return f(pth).predict(smiles).item()
    return scorer


def get_sa_scorer() -> Callable[[str], float]:
    from rdkit import Chem
    from rdkit.Chem import RDConfig
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer

    def scorer(smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        return sascorer.calculateScore(mol)
    return scorer


if __name__ == "__main__":
    ra_scorer = get_ra_scorer("DNN", "chembl")
    sa_scorer = get_sa_scorer()
    sc_scorer = get_sc_scorer()

    def scorer(smiles: str):
        return smiles and {
            "ra": ra_scorer(smiles),
            "sa": sa_scorer(smiles),
            "sc": sc_scorer(smiles),
        }
    serve(scorer)
