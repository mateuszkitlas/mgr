import inspect
import json
from typing import Any, Optional, Tuple, TypeVar

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import spearmanr

from shared import Fn

from .data import data, load_trees
from .score import Score
from .tree import Tree, TreeTypes, sum_tree_stats
from .utils import flatten, serialize_dict

matplotlib.rc("font", size=5)

T = TypeVar("T")


def avg(l: list[float]):
    return sum(l) / len(l)


AggFn = Fn[list[float], float]

# name, color, scatter_kwargs, fn
_agg: list[Tuple[str, str, AggFn]] = [
    ("min", "blue", min),
    ("max", "red", max),
    ("avg", "green", avg),
]

ScoreGetter = Optional[Fn[Score, float]]
_score_getters: list[Tuple[str, ScoreGetter]] = [*Score.getters(), ("ai", None)]


def _unzip(l: list[Tuple[T, T]]):
    return ([x for x, _ in l], [y for _, y in l])


def _all_nodes(roots: list[Tree]):
    return flatten((root.all_nodes() for root in roots))


def _pairs_siblings(roots: list[Tree], xtype: list[TreeTypes], ytype: list[TreeTypes]):
    def pair(node: Tree):
        return (
            (x, y)
            for x in node.children
            if x.type in xtype
            for y in node.children
            if y.type in ytype
        )

    return list(flatten((pair(node) for node in _all_nodes(roots))))


def _pairs_parent_child(
    roots: list[Tree], parenttype: list[TreeTypes], childtype: list[TreeTypes]
):
    def pair(p: Tree):
        return ((p, c) for c in p.children if c.type in childtype)

    return list(
        flatten(
            (pair(parent) for parent in _all_nodes(roots) if parent.type in parenttype)
        )
    )


def display_dict(data: dict[str, Any]):
    serialized = serialize_dict(data, "\n")
    display(HTML(f"<pre>{serialized}</pre>"))


def _ax6(title: dict[str, Any], has_data: bool):
    if has_data:
        display_dict(title)
        fig, axs = plt.subplots(nrows=3, ncols=2)
        fig.subplots_adjust(hspace=0.25, bottom=0.05, left=0, top=0.94)
        fig.set_dpi(220)
        fig.set_figheight(7)
        # fig.suptitle(_title)
        for (score_name, getter), ax in zip(_score_getters, axs.flat):
            ax.set_title(f"score={score_name}")
            ax.ticklabel_format(style="plain")
            yield (score_name, getter, ax)
        plt.show()
    else:
        display_dict({"error": "too little data", **title})


def _hist(ax: Any, x: list[float], bin_count: int, color: str):
    counts, bins, _patches = ax.hist(x=[x], color=[color], bins=bin_count)
    ax.set_xticks(bins)
    bin_centers = 0.5 * np.diff(bins) + bins[:-1]
    for count, x in zip(counts, bin_centers):
        percent = "%0.0f%%" % (100 * float(count) / counts.sum())
        ax.annotate(
            percent,
            xy=(x, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -18),
            textcoords="offset points",
            va="top",
            ha="center",
        )


def input_data(detailed: bool):
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
    for roots, source in input_data(detailed):
        xs, ys = _unzip(_pairs_siblings(roots, xtype, ytype))
        title = {
            "xtype": xtype,
            "ytype": ytype,
            "tree_to_scores": inspect.getsource(tree_to_scores),
        }
        if detailed:
            title["source"] = source
        for agg_name, agg_color, agg_fn in _agg:
            display_dict({"agg_fn": agg_name})
            for score_name, getter, ax in _ax6(title, bool(xs)):

                def stat(t: Tree):
                    return _tree_to_float(t, tree_to_scores, getter, agg_fn)

                x, y = [stat(x) for x in xs], [stat(y) for y in ys]
                rho, pval = spearmanr(x, y)
                info = {
                    "score": score_name,
                    "rho": round(rho, 10),
                    "pval": round(pval, 10),
                }
                _min = min([min(x), min(y)])
                _max = max([max(x), max(y)])
                ax.axis("square")
                ax.set_xlim([_min, _max])
                ax.set_ylim([_min, _max])
                ax.set_title(serialize_dict(info, ", "))
                ax.axline((_min, _min), (_max, _max), linewidth=1, color="black")
                ax.scatter(x, y, s=1, linewidths=1, color=agg_color)


def histogram_pairs_siblings(
    xtype: list[TreeTypes],
    ytype: list[TreeTypes],
    tree_to_scores: Fn[Tree, list[Score]],
    detailed: bool,
):
    for roots, source in input_data(detailed):
        pairs = _pairs_siblings(roots, xtype, ytype)
        title = {
            "xtype": xtype,
            "ytype": ytype,
            "tree_to_scores": inspect.getsource(tree_to_scores),
            "description": "f(xtype) - f(ytype)",
        }
        if detailed:
            title["source"] = source
        for agg_name, agg_color, agg_fn in _agg:
            display_dict({"agg_fn": agg_name})
            for score_name, getter, ax in _ax6(title, bool(pairs)):

                def stat(t: Tree) -> float:
                    return _tree_to_float(t, tree_to_scores, getter, agg_fn)

                if score_name == "mf":
                    ax.xaxis.set_tick_params(labelsize="x-small")
                    ax.xaxis.set_major_formatter(FormatStrFormatter("%0.0f"))
                else:
                    ax.xaxis.set_major_formatter(FormatStrFormatter("%0.1f"))
                _hist(
                    ax,
                    [stat(x) - stat(y) for x, y in pairs],
                    16 if score_name == "mf" else 10,
                    agg_color,
                )


def histogram_pairs_parent_child(
    parenttype: list[TreeTypes],
    childtype: list[TreeTypes],
    tree_to_scores: Fn[Tree, list[Score]],
    detailed: bool,
):
    for roots, source in input_data(detailed):
        pairs = _pairs_parent_child(roots, parenttype, childtype)
        title = {
            "parenttype": parenttype,
            "childtype": childtype,
            "tree_to_scores": inspect.getsource(tree_to_scores),
            "description": "f(parenttype) - f(childtype)",
        }
        if detailed:
            title["source"] = source
        for agg_name, agg_color, agg_fn in _agg:
            display_dict({"agg_fn": agg_name})
            for score_name, getter, ax in _ax6(title, bool(pairs)):

                def stat(t: Tree) -> float:
                    return _tree_to_float(t, tree_to_scores, getter, agg_fn)

                if score_name == "mf":
                    ax.xaxis.set_tick_params(labelsize="x-small")
                    ax.xaxis.set_major_formatter(FormatStrFormatter("%0.0f"))
                else:
                    ax.xaxis.set_major_formatter(FormatStrFormatter("%0.1f"))
                _hist(
                    ax,
                    [stat(x) - stat(y) for x, y in pairs],
                    16 if score_name == "mf" else 10,
                    agg_color,
                )


def boxplot_scores(
    ltype: list[TreeTypes],
    rtype: list[TreeTypes],
    tree_to_scores: Fn[Tree, list[Score]],
    detailed: bool,
):
    for roots, source in input_data(detailed):
        xs = [node for node in _all_nodes(roots) if node.type in ltype]
        ys = [node for node in _all_nodes(roots) if node.type in rtype]
        title = {
            "ltype": ltype,
            "rtype": rtype,
            "tree_to_scores": inspect.getsource(tree_to_scores),
        }
        if detailed:
            title["source"] = source
        for _, getter, ax in _ax6(title, bool(xs) and bool(ys)):
            for i, (agg_name, agg_color, agg_fn) in enumerate(_agg):

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
