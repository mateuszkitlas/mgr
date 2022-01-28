from typing import Tuple
from IPython.display import display

import matplotlib.pyplot as plt
from matplotlib.axis import Axis

from .score import Score, Smiles
from .tree import Tree
from .utils import cast, flatten


def _pairs(node: Tree):
    return ((in_stock, expandable) for in_stock in node.in_stock for expandable in node.expandable)


def _unzip_pairs(l: list[Tuple[Smiles, Smiles]]):
    return ([x for x, _ in l], [y for _, y in l])


def scatterPairs(root: Tree):
    points = list(flatten((_pairs(node) for node in root.all_nodes())))
    xs, ys = _unzip_pairs(points)
    fig, axs = plt.subplots(2, 2)
    fig.subplots_adjust(hspace=0.5)
    fig.set_dpi(300)
    for (name, getter), ax in zip(Score.getters(), cast(list[Axis], axs.flat)):
        x, y = [getter(x.score) for x in xs], [getter(y.score) for y in ys]
        # display((name, x, y))
        ax.set_title(name)
        ax.scatter(x, y)
    plt.show()
