from typing import Any, Callable, Tuple, TypeVar

import matplotlib.pyplot as plt
import matplotlib
from IPython.display import HTML, display

from .data import Mol
from .score import Score
from .tree import Tree
from .utils import flatten

matplotlib.rc('font', size=5)

T = TypeVar("T")


def avg(l: list[float]):
    return sum(l) / len(l)


_funs: list[Tuple[str, str, dict[str, Any], Callable[[list[float]], float]]] = [
    ("min", "blue", {"marker":"o", "edgecolor": "blue", "facecolor": "none"}, min),
    ("max", "red", {"marker": "x", "color": "red"}, max),
    ("avg", "green", {"marker": "+", "color": "green"}, avg),
]
_funs_names = ", ".join((f"{name} ({color} marker)" for name, color, _2, _3 in _funs))

_score_getters = [*Score.getters(), ("ai", None)]

def _pairs(node: Tree):
    return (
        (solvable, not_solvable)
        for solvable in (c for c in node.children if c.solvable and not c.solved)
        for not_solvable in (c for c in node.children if not c.solvable)
    )


def _unzip_pairs(l: list[Tuple[T, T]]):
    return ([x for x, _ in l], [y for _, y in l])


def _scatter_pairs_for_roots(roots: list[Tree], source: str):
    all_nodes = flatten((root.all_nodes() for root in roots))
    points = list(flatten((_pairs(node) for node in all_nodes)))
    if points:
        fig, axs = plt.subplots(nrows=3, ncols=2)
        fig.subplots_adjust(hspace=0.25, bottom=0.05, left=0, top=0.94)
        fig.set_dpi(220)
        fig.set_figheight(7)
        fig.suptitle(f"source={source}")
        for _1, _2, kwargs, stat_fn in _funs:
            xs, ys = _unzip_pairs(points)
            
            for (score_name, getter), ax in zip(_score_getters, axs.flat):

                def stat(t: Tree) -> float:
                    if getter:
                        return stat_fn([getter(s.score) for s in t.expandable])
                    else:
                        return t.ai_score

                x, y = [stat(x) for x in xs], [stat(y) for y in ys]
                ax.set_title(f"score={score_name}")
                ax.scatter(x, y, s=4, linewidths=1, **kwargs)
                ax.ticklabel_format(style='plain')
        plt.show()
    else:
        display(HTML(f"too little data in source={source}"))


_score_names = ", ".join((name for name, _ in _score_getters))

_scatter_pairs_desc = f"""
<span style='font-family:"Courier New"'>source, stat_fn, score</span> - displayed on figure
<pre>
for every node in source:
  for every (x,y) in node.children:
    where
      x = solvable inner node, represended on x-axis
      y = not solvable node, represented on y-axis
    for score in [{_score_names}]:
      # if score==ai: avg == min == max
      for stat_fn in [{_funs_names}]:
        point = fun(x), fun(y)
        where
          fun(node) = stat_fn([score(mol) for mol in node.expandable])
          node.expandable are not solved molecules (not available in stock)
</pre>
"""


def scatter_pairs(mols: list[Tuple[Mol, Tree]]):
    display(HTML(_scatter_pairs_desc))
    solvable_mols = len([None for _, root in mols if root.solvable])
    _scatter_pairs_for_roots([root for _, root in mols], f"all ({solvable_mols} solvable mols)")
    for mol, root in mols:
        _scatter_pairs_for_roots([root], f"{mol.name}; {root.stats()}")
