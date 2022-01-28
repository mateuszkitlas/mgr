from typing import Iterable, Tuple

import matplotlib.pyplot as plt

from .score import Score, Smiles
from .tree import Tree
from .utils import flatten


def pairs(node: Tree):
    for in_stock in node.in_stock:
        for expandable in node.expandable:
            yield (in_stock, expandable)


def unzip_pairs(l: Iterable[Tuple[Smiles, Smiles]]):
    return ((x for x, _ in l), (y for _, y in l))


def stats1(root: Tree):
    points = flatten((pairs(node) for node in root.all_nodes()))
    xs, ys = unzip_pairs(points)
    for name, getter in Score.getters():
        x, y = (getter(x.score) for x in xs), (getter(y.score) for y in ys)
        plt.scatter(list(x), list(y))
        plt.show()
