import inspect
from typing import Any, Optional, Tuple, TypeVar

import matplotlib
import matplotlib.pyplot as plt
from IPython.display import HTML, display

from shared import Fn

from .data import data, load_trees
from .score import Score
from .tree import Tree, TreeTypes, sum_tree_stats
from .utils import flatten, serialize_dict
import json

matplotlib.rc("font", size=5)

T = TypeVar("T")


def avg(l: list[float]):
    return sum(l) / len(l)


AggFn = Fn[list[float], float]

# name, color, scatter_kwargs, fn
_agg: list[Tuple[str, str, dict[str, Any], AggFn]] = [
    ("min", "blue", {"marker": "o", "edgecolor": "blue", "facecolor": "none"}, min),
    ("max", "red", {"marker": "x", "color": "red"}, max),
    ("avg", "green", {"marker": "+", "color": "green"}, avg),
]

ScoreGetter = Optional[Fn[Score, float]]
_score_getters: list[Tuple[str, ScoreGetter]] = [*Score.getters(), ("ai", None)]


def _unzip(l: list[Tuple[T, T]]):
    return ([x for x, _ in l], [y for _, y in l])


def _all_nodes(roots: list[Tree]):
    return flatten((root.all_nodes() for root in roots))


def _pairs(roots: list[Tree], xtype: list[TreeTypes], ytype: list[TreeTypes]):
    def pair_children(node: Tree):
        return (
            (x, y)
            for x in node.children
            if x.type in xtype
            for y in node.children
            if y.type in ytype
        )

    return list(flatten((pair_children(node) for node in _all_nodes(roots))))


def _print(data: dict[str, Any]):
    serialized = serialize_dict(data, "\n")
    display(HTML(f"<pre>{serialized}</pre>"))


def _ax6(title: dict[str, Any], has_data: bool):
    if has_data:
        _print(title)
        fig, axs = plt.subplots(nrows=3, ncols=2)
        fig.subplots_adjust(hspace=0.25, bottom=0.05, left=0, top=0.94)
        fig.set_dpi(220)
        fig.set_figheight(7)
        # fig.suptitle(_title)
        for (score_name, getter), ax in zip(_score_getters, axs.flat):
            ax.set_title(f"score={score_name}")
            ax.ticklabel_format(style="plain")
            yield (getter, ax)
        plt.show()
    else:
        _print({"error": "too little data", **title})


def _data(detailed: bool):
    mol_by_smiles = {mol.smiles: mol for mol in data()}
    mols = [
        (mol_by_smiles[smiles], Tree(json_tree))
        for (smiles, json_tree) in load_trees("trees.json")
    ]
    solved_count = sum((root.type == "internal" for _, root in mols))
    not_solved_count = sum((root.type == "not_solved" for _, root in mols))
    all_count = len(mols)
    assert sum((root.type == "solved" for _, root in mols)) == 0
    assert (solved_count + not_solved_count) == all_count
    sum_stats = sum_tree_stats([root.stats() for _, root in mols])
    yield (
        [root for _, root in mols],
        f"ALL ({solved_count}/{all_count} solved); {json.dumps(sum_stats, indent=2)}",
    )
    if detailed:
        for mol, root in mols:
            yield ([root], f"{mol.name}; {json.dumps(root.stats(), indent=2)}")


def _tree_to_float(
    tree: Tree,
    tree_to_scores: Fn[Tree, list[Score]],
    getter: ScoreGetter,
    agg_fn: AggFn,
) -> float:
    if getter:
        return agg_fn([getter(score) for score in tree_to_scores(tree)])
    else:
        return tree.ai_score


def scatter_pairs(
    xtype: list[TreeTypes],
    ytype: list[TreeTypes],
    tree_to_scores: Fn[Tree, list[Score]],
    detailed: bool,
):
    for roots, source in _data(detailed):
        xs, ys = _unzip(_pairs(roots, xtype, ytype))
        title = {
            "source": source,
            "xtype": xtype,
            "ytype": ytype,
            "tree_to_scores": inspect.getsource(tree_to_scores),
        }
        for getter, ax in _ax6(title, bool(xs)):
            for _1, _2, agg_scatter_kwargs, agg_fn in _agg:

                def stat(t: Tree):
                    return _tree_to_float(t, tree_to_scores, getter, agg_fn)

                x, y = [stat(x) for x in xs], [stat(y) for y in ys]
                ax.scatter(x, y, s=4, linewidths=1, **agg_scatter_kwargs)


def histogram_pairs(
    xtype: list[TreeTypes],
    ytype: list[TreeTypes],
    tree_to_scores: Fn[Tree, list[Score]],
    detailed: bool,
):
    for roots, source in _data(detailed):
        pairs = _pairs(roots, xtype, ytype)
        title = {
            "source": source,
            "xtype": xtype,
            "ytype": ytype,
            "tree_to_scores": inspect.getsource(tree_to_scores),
        }
        for getter, ax in _ax6(title, bool(pairs)):

            def stat(t: Tree, agg_fn: AggFn) -> float:
                return _tree_to_float(t, tree_to_scores, getter, agg_fn)

            ax.hist(
                x=[
                    [stat(x, agg_fn) - stat(y, agg_fn) for x, y in pairs]
                    for _1, _2, _3, agg_fn in _agg
                ],
                color=[agg_color for _1, agg_color, _2, _3 in _agg],
            )


def boxplot_scores(
    ltype: list[TreeTypes],
    rtype: list[TreeTypes],
    tree_to_scores: Fn[Tree, list[Score]],
    detailed: bool,
):
    for roots, source in _data(detailed):
        xs = [node for node in _all_nodes(roots) if node.type in ltype]
        ys = [node for node in _all_nodes(roots) if node.type in rtype]
        title = {
            "source": source,
            "ltype": ltype,
            "rtype": rtype,
            "tree_to_scores": inspect.getsource(tree_to_scores),
        }
        for getter, ax in _ax6(title, bool(xs) and bool(ys)):
            for i, (agg_name, agg_color, _3, agg_fn) in enumerate(_agg):

                def stat(t: Tree) -> float:
                    return _tree_to_float(t, tree_to_scores, getter, agg_fn)

                x, y = [stat(x) for x in xs], [stat(y) for y in ys]

                def label(sublabel: str):
                    return agg_name + (f"\n{sublabel}" if i == 1 else "")

                ax.boxplot(
                    [x, y],
                    labels=[label(str(ltype)), label(str(rtype))],
                    positions=[i, i + len(_agg)],
                    boxprops=dict(color=agg_color),
                    capprops=dict(color=agg_color),
                    whiskerprops=dict(color=agg_color),
                    flierprops=dict(color=agg_color, markeredgecolor=agg_color),
                    medianprops=dict(color=agg_color),
                )
