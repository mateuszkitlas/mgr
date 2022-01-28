from typing import Tuple

import matplotlib.pyplot as plt
from IPython.display import display
from matplotlib.axis import Axis

from .data import Mol
from .score import Score, Smiles
from .tree import Tree
from .utils import cast, flatten


def _pairs(node: Tree):
    return (
        (in_stock, expandable)
        for in_stock in node.in_stock
        for expandable in node.expandable
    )


def _unzip_pairs(l: list[Tuple[Smiles, Smiles]]):
    return ([x for x, _ in l], [y for _, y in l])


def _scatter_pairs_for_roots(roots: list[Tree], text: str):
    all_nodes = flatten((root.all_nodes() for root in roots))
    points = list(flatten((_pairs(node) for node in all_nodes)))
    if points:
        xs, ys = _unzip_pairs(points)
        fig, axs = plt.subplots(2, 2)
        fig.subplots_adjust(hspace=0.5)
        fig.set_dpi(300)
        fig.text(0, 0, text)
        for (name, getter), ax in zip(Score.getters(), cast(list[Axis], axs.flat)):
            x, y = [getter(x.score) for x in xs], [getter(y.score) for y in ys]
            ax.set_title(name)
            ax.scatter(x, y)
        plt.show()
    else:
        display("No points")


def _scatter_pairs_for_mol(mol: Mol, root: Tree, text: str):
    display(str(mol), root.stats())
    _scatter_pairs_for_roots([root], text)


def scatter_pairs(mols: list[Tuple[Mol, Tree]]):
    _scatter_pairs_for_roots([root for _, root in mols], "aaaaaa\nbbbb\ncccc")
    for mol, root in mols:
        _scatter_pairs_for_mol(mol, root, "ss")
