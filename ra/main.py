from typing import Callable, Optional
from shared import serve, project_dir


def get_scorer(
    model: str,  # Literal["DNN", "XGB"],
    db: str,  # Literal["chembl", "gdbchembl", "gdbmedchem"]
) -> Callable[[str], Optional[float]]:
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

    def scorer(smiles: str) -> Optional[float]:
        if smiles:
            return f(pth).predict(smiles).item()
    return scorer


if __name__ == "__main__":
    scorer = get_scorer("DNN", "chembl")
    serve(scorer)
