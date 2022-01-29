from typing import Callable, Tuple, TypeVar

import matplotlib.pyplot as plt
from IPython.display import HTML, display
from matplotlib.axis import Axis

from .data import Mol
from .score import Score
from .tree import Tree
from .utils import cast, flatten

T = TypeVar("T")


def avg(l: list[float]):
    return sum(l) / len(l)


funs: list[Tuple[str, Callable[[list[float]], float]]] = [
    ("min", min),
    ("max", max),
    ("avg", avg),
]


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
        for stat_name, stat_fn in funs:
            xs, ys = _unzip_pairs(points)
            fig, axs = plt.subplots(2, 2)
            fig.subplots_adjust(hspace=0.5, bottom=0.15)
            fig.set_dpi(200)
            fig.text(0, 0, f"stat_fn={stat_name}\nsource={source}")
            for (score_name, getter), ax in zip(
                Score.getters(), cast(list[Axis], axs.flat)
            ):

                def stat(t: Tree) -> float:
                    return stat_fn([getter(s.score) for s in t.expandable])

                x, y = [stat(x) for x in xs], [stat(y) for y in ys]
                ax.set_title(f"score={score_name}")
                ax.scatter(x, y)
            plt.show()
    else:
        display(HTML(f"too little data in source={source}"))


scatter_pairs_desc = """
<span style='font-family:"Courier New"'>source, stat_fn, score</span> - displayed on figure
<pre>
for every node in source:
  for every node.children pair (x,y):
    where
      x = solvable inner node, represended on x-axis
      y = not solvable node, represented on y-axis
    for score in [sa, sc, ra, syba]:
      for stat_fn in [min, max, avg]:
        point = fun(x), fun(y)
        where
          fun(node) = stat_fn([score(mol) for mol in node.expandable])
          node.expandable are not solved molecules (not available in stock)
</pre>
"""


def scatter_pairs(mols: list[Tuple[Mol, Tree]]):
    display(HTML(scatter_pairs_desc))
    _scatter_pairs_for_roots([root for _, root in mols], "all")
    for mol, root in mols:
        _scatter_pairs_for_roots([root], f"{mol.name}; {root.stats()}")
