from typing import List, Optional, Tuple

from shared import serve

from .utils import get_mf_scorer, get_sa_scorer, get_sc_scorer

# from .scscore_tensorflow import get_sc_scorer


if __name__ == "__main__":
    score = {
        "sa": get_sa_scorer(),
        "sc": get_sc_scorer(),
        "mf": get_mf_scorer(),
    }

    def scorer(data: Optional[Tuple[str, List[str]]]):
        if data:
            type, smileses = data
            return [score[type](s) for s in smileses]

    serve(scorer)
