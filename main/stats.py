import inspect
import json
from typing import Any, Optional, Tuple, TypeVar

import matplotlib
import matplotlib.pyplot as plt
from IPython.display import HTML, display
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import spearmanr
from sklearn import metrics

from main.data import Mol
from shared import Db, Fn

from .ai import ai_input_gen
from .score import Score
from .tree import DictAiTree, AiTree, TreeTypes, sum_tree_stats
from .types import AiInput
from .utils import flatten, not_none

matplotlib.rc("font", size=5)

T = TypeVar("T")


def avg(l: list[float]):
    return sum(l) / len(l)


AggFn = Fn[list[float], float]
AggTuple = Tuple[str, str, AggFn]

# name, color, scatter_kwargs, fn
agg_list: list[AggTuple] = [
    ("min", "blue", min),
    ("max", "red", max),
    ("avg", "green", avg),
]

ScoreGetter = Fn[Score, float]


def _unzip(l: list[Tuple[T, T]]):
    return ([x for x, _ in l], [y for _, y in l])


def _all_nodes(roots: list[AiTree]):
    return flatten((root.all_nodes() for root in roots))


def _pairs_siblings(
    roots: list[AiTree], xtype: list[TreeTypes], ytype: list[TreeTypes]
):
    def pair(node: AiTree):
        return (
            (x, y)
            for x in node.children
            if x.type in xtype
            for y in node.children
            if y.type in ytype
        )

    return list(flatten((pair(node) for node in _all_nodes(roots))))


def _pairs_parent_child(
    roots: list[AiTree], parenttype: list[TreeTypes], childtype: list[TreeTypes]
):
    def pair(p: AiTree):
        return ((p, c) for c in p.children if c.type in childtype)

    return list(
        flatten(
            (pair(parent) for parent in _all_nodes(roots) if parent.type in parenttype)
        )
    )


def _fn_txt(fn: Fn[Any, Any]):
    return inspect.getsource(fn).replace("\n", "")


def display_str(data: str):
    display(HTML(f"<pre>{data}</pre>"))


def _ax6(title: str, has_data: bool):
    if has_data:
        fig, axs = plt.subplots(nrows=3, ncols=2)
        fig.subplots_adjust(hspace=0.25, bottom=0.05, left=0, top=0.75)
        fig.set_dpi(220)
        fig.set_figheight(7)
        fig.suptitle(title, horizontalalignment="left", x=0.005)
        for (score_name, getter), ax in zip(Score.getters(), axs.flat):
            ax.set_title(f"score={score_name}")
            ax.ticklabel_format(style="plain")
            yield (score_name, getter, ax)
        plt.show()
    else:
        display_str(f"too little data for: \n{title}")


def _ax2(title: str, ttype: list[TreeTypes], btype: list[TreeTypes]):
    fig, axs = plt.subplots(nrows=2, ncols=1)
    fig.subplots_adjust(hspace=0.25, bottom=0.05, left=0, top=0.75)
    fig.set_dpi(220)
    fig.set_figheight(7)
    fig.suptitle(title, horizontalalignment="left", x=0.005)
    for type, ax in zip([ttype, btype], axs.flat):
        ax.set_title(f"type={type}")
        ax.ticklabel_format(style="plain")
        yield (type, ax)
    plt.show()


def _hist(
    ax: Any,
    x: list[float],
    bin_count: int,
    color: str,
    range: Optional[Tuple[float, float]] = None,
):
    return ax.hist(x=[x], color=[color], bins=bin_count, range=range)
    # ax.set_xticks(bins)
    """
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
    """


def input_data(detailed: bool):
    def f(db: Db, db_scores_merged: Db, ai_input: AiInput, mol: Mol):
        return (
            AiTree(
                (db.read(["ai_postprocess", ai_input], DictAiTree), db_scores_merged)
            ),
            mol,
        )

    with Db("scores_merged", True) as db_scores_merged:
        data = list(
            not_none(
                f(db, db_scores_merged, ai_input, mol)
                for db, ai_input, mol in ai_input_gen(True, True)
            )
        )
    solved_count = sum((root.type == "internal" for root, _mol in data))
    not_solved_count = sum((root.type == "not_solved" for root, _mol in data))
    all_count = len(data)
    assert sum((root.type == "solved" for root, _mol in data)) == 0
    assert (solved_count + not_solved_count) == all_count
    sum_stats = sum_tree_stats([root.stats() for root, _mol in data])
    yield (
        [root for root, _mol in data],
        f"ALL ({solved_count}/{all_count} solved); {json.dumps(sum_stats, indent=2)}",
    )
    if detailed:
        for root, mol in data:
            yield ([root], f"{mol.name}; {json.dumps(root.stats(), indent=2)}")


def _tree_to_float(
    tree: AiTree,
    tree_to_scores: Fn[AiTree, list[Score]],
    getter: ScoreGetter,
    agg_fn: AggFn,
) -> float:
    return agg_fn([getter(score) for score in tree_to_scores(tree)])


def scatter_pairs(
    xtype: list[TreeTypes],
    ytype: list[TreeTypes],
    tree_to_scores: Fn[AiTree, list[Score]],
    detailed: bool,
):
    for roots, source in input_data(detailed):
        if detailed:
            display_str(source)
        xs, ys = _unzip(_pairs_siblings(roots, xtype, ytype))
        for agg_name, agg_color, agg_fn in agg_list:
            title = f"""
# scatter plot of x_y_pairs

tree_to_scores = {_fn_txt(tree_to_scores)}

for every tree in source:
  for every node in tree:
    for every pair (xnode, ynode) in node.children:
      if xnode.type in {xtype} and ynode.type in {ytype}:
        x_y_pairs.append((
          {agg_name}(tree_to_scores(xnode)),
          {agg_name}(tree_to_scores(ynode))
        ))"""
            for score_name, getter, ax in _ax6(title, bool(xs)):

                def stat(t: AiTree):
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
                ax.set_title(info)
                ax.axline((_min, _min), (_max, _max), linewidth=1, color="black")
                ax.scatter(x, y, s=1, linewidths=1, color=agg_color)


def histogram_pairs_siblings(
    xtype: list[TreeTypes],
    ytype: list[TreeTypes],
    tree_to_scores: Fn[AiTree, list[Score]],
    detailed: bool,
):
    for roots, source in input_data(detailed):
        if detailed:
            display_str(source)
        pairs = _pairs_siblings(roots, xtype, ytype)
        for agg_name, agg_color, agg_fn in agg_list:
            title = f"""
# histogram of x

tree_to_scores = {_fn_txt(tree_to_scores)}

for every tree in source:
  for every node in tree:
    for every pair (xnode, ynode) in node.children:
      if xnode.type in {xtype} and ynode.type in {ytype}:
        x.append({agg_name}(tree_to_scores(xnode)) - {agg_name}(tree_to_scores(ynode)))"""
            for _score_name, getter, ax in _ax6(title, bool(pairs)):

                def stat(t: AiTree) -> float:
                    return _tree_to_float(t, tree_to_scores, getter, agg_fn)

                ax.xaxis.set_tick_params(labelsize="x-small")
                ax.xaxis.set_major_formatter(FormatStrFormatter("%0.0f"))
                _hist(
                    ax,
                    [stat(x) - stat(y) for x, y in pairs],
                    20,  # 16 if score_name == "mf" else 10,
                    agg_color,
                )
                # np.linspace(from, to, step)


def histogram_pairs_parent_child(
    parenttype: list[TreeTypes],
    childtype: list[TreeTypes],
    tree_to_scores: Fn[AiTree, list[Score]],
    detailed: bool,
):
    for roots, source in input_data(detailed):
        if detailed:
            display_str(source)
        pairs = _pairs_parent_child(roots, parenttype, childtype)
        for agg_name, agg_color, agg_fn in agg_list:
            title = f"""
# histogram of x

tree_to_scores = {_fn_txt(tree_to_scores)}

for every tree in source:
  for every node in tree:
    for every child_node in node.children:
      if node.type in {parenttype} and child_node in {childtype}:
        x.append({agg_name}(tree_to_scores(node)) - {agg_name}(tree_to_scores(child_node)))"""
            for _score_name, getter, ax in _ax6(title, bool(pairs)):

                def stat(t: AiTree) -> float:
                    return _tree_to_float(t, tree_to_scores, getter, agg_fn)

                ax.xaxis.set_tick_params(labelsize="x-small")
                # ax.xaxis.set_major_formatter(FormatStrFormatter("%0.0f"))
                _hist(
                    ax,
                    [stat(x) - stat(y) for x, y in pairs],
                    40,  # 16 if score_name == "mf" else 10,
                    agg_color,
                )


def boxplot_scores(
    ltype: list[TreeTypes],
    rtype: list[TreeTypes],
    tree_to_scores: Fn[AiTree, list[Score]],
    detailed: bool,
):
    for roots, source in input_data(detailed):
        if detailed:
            display_str(source)
        xs = [node for node in _all_nodes(roots) if node.type in ltype]
        ys = [node for node in _all_nodes(roots) if node.type in rtype]
        title = f"""
# boxplots of left_min, left_max, left_avg, right_min, right_max, right_avg

tree_to_scores = {_fn_txt(tree_to_scores)}

for every tree in source:
  for every node in tree:
    if node.type in {ltype}:
      left_min.append(min(tree_to_scores(node)))
      left_max.append(max(tree_to_scores(node)))
      left_avg.append(avg(tree_to_scores(node)))
    if node.type in {rtype}:
      right_min.append(min(tree_to_scores(node)))
      right_max.append(max(tree_to_scores(node)))
      right_avg.append(avg(tree_to_scores(node)))"""
        for _, getter, ax in _ax6(title, bool(xs) and bool(ys)):
            for i, (agg_name, agg_color, agg_fn) in enumerate(agg_list):

                def stat(t: AiTree) -> float:
                    return _tree_to_float(t, tree_to_scores, getter, agg_fn)

                x, y = [stat(x) for x in xs], [stat(y) for y in ys]

                def label(sublabel: str):
                    return agg_name + (f"\n{sublabel}" if i == 1 else "")

                ax.boxplot(
                    [x, y],
                    labels=[label(str(ltype)), label(str(rtype))],
                    positions=[i, i + len(agg_list)],
                    boxprops=dict(color=agg_color),
                    capprops=dict(color=agg_color),
                    whiskerprops=dict(color=agg_color),
                    flierprops=dict(color=agg_color, markeredgecolor=agg_color),
                    medianprops=dict(color=agg_color),
                )


def histogram_top_bottom(
    ttype: list[TreeTypes],
    btype: list[TreeTypes],
    tree_to_scores: Fn[AiTree, list[Score]],
    agg_tuple: AggTuple,
    score_getter: Tuple[str, ScoreGetter],
    detailed: bool,
):
    for roots, _source in input_data(detailed):
        agg_name, agg_color, agg_fn = agg_tuple
        score_name, getter = score_getter

        def stat(t: AiTree) -> float:
            return _tree_to_float(t, tree_to_scores, getter, agg_fn)

        title = f"""
# histogram of top, bottom

tree_to_scores = {_fn_txt(tree_to_scores)}
score = {score_name}

for every tree in source:
  for every node in tree:
    if node.type in {ttype}:
      top.append({agg_name}(tree_to_scores(node))))
    if node.type in {btype}:
      top.append({agg_name}(tree_to_scores(node))))
"""
        all_nodes = list(flatten((root.all_nodes() for root in roots)))

        def minmax(l: list[float]):
            return min(l), max(l)

        range = minmax([stat(tree) for tree in all_nodes])
        for type, ax in _ax2(title, ttype, btype):
            # ax.set_ylim((0, 3000))
            _hist(
                ax,
                [stat(tree) for tree in all_nodes if tree.type in type],
                30,
                agg_color,
                range=range,
            )
        y = [1 for tree in all_nodes if tree.type in ttype] + [
            0 for tree in all_nodes if tree.type in btype
        ]
        pred = [stat(tree) for tree in all_nodes if tree.type in ttype] + [
            stat(tree) for tree in all_nodes if tree.type in btype
        ]
        fpr, tpr, _thresholds = metrics.roc_curve(y, pred)
        roc_auc = metrics.auc(fpr, tpr)
        display = metrics.RocCurveDisplay(
            fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=f"{score_name} {agg_name}"
        )

        fig, ax = plt.subplots()
        fig.subplots_adjust(hspace=0.25, bottom=0.05, left=0, top=0.75)
        fig.set_dpi(220)
        fig.set_figheight(7)
        display.plot(ax)
        plt.show()
