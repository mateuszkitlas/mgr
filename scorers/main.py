from shared import Fn, Timer, paracetamol_smiles, serve

from .utils import (get_mf_scorer, get_ra_scorer, get_sa_scorer, get_sc_scorer,
                    get_syba_scorer)

# from .scscore_tensorflow import get_sc_scorer

Scorer = Fn[str, float]


def j(ra: float, sa: float, sc: float, mf: float, syba: float):
    return {"ra": ra, "sa": sa, "sc": sc, "mf": mf, "syba": syba}


if __name__ == "__main__":
    ra_time, ra_scorer = Timer.calc(lambda: get_ra_scorer("DNN", "chembl"))
    sa_time, sa_scorer = Timer.calc(get_sa_scorer)
    sc_time, sc_scorer = Timer.calc(get_sc_scorer)
    mf_time, mf_scorer = Timer.calc(get_mf_scorer)
    syba_time, syba_scorer = Timer.calc(get_syba_scorer)

    times = j(ra=ra_time, sa=sa_time, sc=sc_time, mf=mf_time, syba=syba_time)
    print(f"Loading times: {times}")

    def scorer(smiles: str):
        if smiles:
            tra, ra = Timer.calc(lambda: ra_scorer(smiles))
            tsa, sa = Timer.calc(lambda: sa_scorer(smiles))
            tsc, sc = Timer.calc(lambda: sc_scorer(smiles))
            tsyba, syba = Timer.calc(lambda: syba_scorer(smiles))
            tmf, mf = Timer.calc(lambda: mf_scorer(smiles))
            return (
                j(ra=tra, sa=tsa, sc=tsc, mf=tmf, syba=tsyba),
                j(ra=ra, sa=sa, sc=sc, mf=mf, syba=syba),
            )

    scorer(paracetamol_smiles)  # warm up

    serve(scorer)
