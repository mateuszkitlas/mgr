from typing import Any, Tuple, TypeVar, Union

import matplotlib
import matplotlib.pyplot as plt
from IPython.display import HTML, display

from shared import Fn

from .data import Mol, data, load_trees
from .score import Score
from .tree import Tree, TreeTypes
from .utils import flatten, serialize_dict

matplotlib.rc("font", size=5)

T = TypeVar("T")


def avg(l: list[float]):
    return sum(l) / len(l)


def _span(txt: str, styles: dict[str, Union[str, int]]):
    return f"<span style='{serialize_dict(styles, ';')}'>{txt}</span>"


_funs: list[Tuple[str, str, dict[str, Any], Fn[list[float], float]]] = [
    ("min", "blue", {"marker": "o", "edgecolor": "blue", "facecolor": "none"}, min),
    ("max", "red", {"marker": "x", "color": "red"}, max),
    ("avg", "green", {"marker": "+", "color": "green"}, avg),
]
_funs_names = ", ".join(
    (_span(name, {"color": color}) for name, color, _2, _3 in _funs)
)

_score_getters = [*Score.getters(), ("ai", None)]


def _pairs(node: Tree, type1: TreeTypes, type2: TreeTypes):
    return (
        (solvable, not_solvable)
        for solvable in (c for c in node.children if c.type == type1)
        for not_solvable in (c for c in node.children if not c.type == type2)
    )


def _unzip_pairs(l: list[Tuple[T, T]]):
    return ([x for x, _ in l], [y for _, y in l])


_score_names = ", ".join((name for name, _ in _score_getters))


def _pairs_desc(point_desc: str, x_desc: str, y_desc: str):
    return f"""
{_span("source, stat_fn, score", {"font-family": "monospace"})} - displayed on figure
<pre>
for every node in source:
  for every (x,y) in node.children:
    where
      x = {x_desc}, represended on x-axis
      y = {y_desc}, represented on y-axis
    for score in [{_score_names}]:
      # if score==ai: avg == min == max
      for stat_fn in [{_funs_names}]:
        point = {point_desc}
        where
          fun(node) = stat_fn([score(mol) for mol in node.expandable])
          node.expandable are not solved molecules (not available in stock)
</pre>
"""


def _scatter_pairs_for_roots(roots: list[Tree], source: str):
    all_nodes = flatten((root.all_nodes() for root in roots))
    points = list(
        flatten((_pairs(node, "internal", "not_solved") for node in all_nodes))
    )
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
                ax.ticklabel_format(style="plain")
        plt.show()
    else:
        display(HTML(f"too little data in source={source}"))


def scatter_pairs(mols: list[Tuple[Mol, Tree]]):
    display(
        HTML(_pairs_desc("(f(x), f(y))", "solvable inner node", "not solvable node"))
    )
    solvable_mols = len([root for _, root in mols if root.type == "internal"])
    _scatter_pairs_for_roots(
        [root for _, root in mols], f"all ({solvable_mols} solvable mols)"
    )
    for mol, root in mols:
        _scatter_pairs_for_roots(
            [root], f"{mol.name}; {serialize_dict(root.stats, ', ')}"
        )


def _histogram_pairs_for_roots(roots: list[Tree], source: str):
    all_nodes = flatten((root.all_nodes() for root in roots))
    points = list(
        flatten((_pairs(node, "internal", "not_solved") for node in all_nodes))
    )
    if points:
        fig, axs = plt.subplots(nrows=3, ncols=2)
        fig.subplots_adjust(hspace=0.25, bottom=0.05, left=0, top=0.94)
        fig.set_dpi(220)
        fig.set_figheight(7)
        fig.suptitle(f"source={source}")
        for (score_name, getter), ax in zip(_score_getters, axs.flat):

            def stat(t: Tree, stat_fn: Fn[list[float], float]) -> float:
                if getter:
                    return stat_fn([getter(s.score) for s in t.expandable])
                else:
                    return t.ai_score

            ax.set_title(f"score={score_name}")
            ax.hist(
                x=[
                    [stat(x, stat_fn) - stat(y, stat_fn) for x, y in points]
                    for _1, _2, _3, stat_fn in _funs
                ],
                color=[color for _1, color, _2, _3 in _funs],
            )
            ax.ticklabel_format(style="plain")
        plt.show()
    else:
        display(HTML(f"too little data in source={source}"))


def histogram_pairs(mols: list[Tuple[Mol, Tree]]):
    display(
        HTML(_pairs_desc("f(x) - f(y)", "solvable inner node", "not solvable node"))
    )
    solvable_mols = len([root for _, root in mols if root.type == "internal"])
    _histogram_pairs_for_roots(
        [root for _, root in mols], f"all ({solvable_mols} solvable mols)"
    )
    for mol, root in mols:
        _histogram_pairs_for_roots(
            [root], f"{mol.name}; {serialize_dict(root.stats, ', ')}"
        )


def main(fn: Fn[list[Tuple[Mol, Tree]], None]):
    mol_by_smiles = {mol.smiles: mol for mol in data()}
    mols = [
        (mol_by_smiles[smiles], Tree(json_tree))
        for (smiles, json_tree) in load_trees("trees.json")
    ]
    fn(mols)
