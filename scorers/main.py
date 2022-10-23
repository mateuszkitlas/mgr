import os
import sys
from typing import Dict, List, NewType, Optional, Tuple, TypeVar

from rdkit import Chem

from shared import Db, Fn, paracetamol_smiles, project_dir, serve

Mol = NewType("Mol", object)
T = TypeVar("T")
SmilesScorer = Fn[str, float]
MolScorer = Fn[Mol, float]

to_mol: Fn[str, Mol] = Chem.MolFromSmiles
to_smiles: Fn[Mol, str] = Chem.MolToSmiles
inverted: Fn[Fn[float, float], Fn[float, float]] = lambda f: lambda value: 1.0 - f(
    value
)


def wrap_to_mol(f: Fn[Mol, T]) -> Fn[str, T]:
    return lambda smiles: f(to_mol(smiles))


def scaler(lowest: float, highest: float) -> Fn[float, float]:
    """
    Returns scale function that transforms linearly value, where value in [lowest, highest], to [0,1]
    0 means infeasible (not-accessible) molecule
    1 means fully feasible molecule
    """
    size = highest - lowest
    return lambda value: (min(max(value, lowest), highest) - lowest) / size


def get_mf_scorer() -> SmilesScorer:
    from .mf_score.mfscore.ocsvm import score_smiles

    scale = scaler(-800, 600)
    return lambda smiles: scale(score_smiles(smiles))


def get_sc_scorer() -> MolScorer:
    from .scscore_numpy import SCScorer

    sc = SCScorer()
    sc.restore()
    scale = inverted(scaler(1, 5))
    return lambda mol: scale(sc.apply(sc.mol_to_fp(sc, mol)))


def get_syba_scorer() -> MolScorer:
    print("Loading syba scorer. It's gonna take ~2 minutes.")
    from syba.syba import SybaClassifier

    syba = SybaClassifier()
    syba.fitDefaultScore()
    scale = scaler(-100, 100)
    return lambda mol: scale(syba.predict(mol=mol))


def get_ra_scorer(
    model: str,  # Literal["DNN", "XGB"],
    db: str,  # Literal["chembl", "gdbchembl", "gdbmedchem"]
) -> SmilesScorer:
    from RAscore import RAscore_NN, RAscore_XGB

    dnn = model == "DNN"
    x = "fcfp" if dnn else "ecfp"
    y = "h5" if dnn else "pkl"
    model = (RAscore_NN.RAScorerNN if dnn else RAscore_XGB.RAScorerXGB)(
        f"{project_dir}/data/ra_models/{model}_{db}_{x}_counts/model.{y}"
    )
    scale = scaler(0, 1)
    return lambda smiles: scale(model.predict(smiles).item())


def get_sa_scorer() -> MolScorer:
    from rdkit.Chem import RDConfig

    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer

    scale = inverted(scaler(1, 10))
    return lambda mol: scale(sascorer.calculateScore(mol))


if __name__ == "__main__":
    scorers: Dict[str, SmilesScorer] = {
        "ra": get_ra_scorer("DNN", "chembl"),
        "sa": wrap_to_mol(get_sa_scorer()),
        "sc": wrap_to_mol(get_sc_scorer()),
        "mf": get_mf_scorer(),
        "syba": wrap_to_mol(get_syba_scorer()),
    }
    smileser = wrap_to_mol(to_smiles)
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
