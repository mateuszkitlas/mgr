from asyncio import gather
from itertools import chain
from typing import (Awaitable, Callable, Iterable, Optional, TypedDict,
                    TypeVar, Union)

from .score import JsonSmiles, Smiles
from .types import AiTree
from .utils import flatten

T = TypeVar("T")
Scorer = Callable[[str], Awaitable[Smiles]]


class _Tree:
    """
    Turns nodes like this:
    ```py
    AiTree(solved=False, children=[
        AiTree(solved=False),
        AiTree(solved=False),
        AiTree(solved=False),
    ])
    ```
    To one node:
    ```py
    _Tree(solved=False, solvable=False, children=[])
    ```
    """

    def __init__(self, ai: AiTree):
        self.solved = ai["is_solved"]
        self.score = ai["score"]
        self._expandable_smiles = ai["expandable_smiles"]
        self._in_stock_smiles = ai["in_stock_smiles"]

        true_children = (c for c in ai["children"] if c)
        compressed_children = [_Tree(c) for c in true_children]

        self.solvable = self.solved or any(c for c in compressed_children if c.solvable)
        self.children = compressed_children if self.solvable else []
        self.expandable: Optional[list[Smiles]] = None
        self.in_stock: Optional[list[Smiles]] = None

    async def _assign_scores(self, f: Scorer):
        assert self.expandable is None
        assert self.in_stock is None
        expandable, in_stock = await gather(
            gather(*(f(e) for e in self._expandable_smiles)),
            gather(*(f(e) for e in self._in_stock_smiles)),
        )
        self.expandable, self.in_stock = list(expandable), list(in_stock)

    def all_nodes(self) -> Iterable["_Tree"]:
        return chain([self], flatten((c.all_nodes() for c in self.children)))

    async def assign_scores_rec(self, f: Scorer):
        await gather(*(t._assign_scores(f) for t in self.all_nodes()))


class JsonTree(TypedDict):
    expandable: list[JsonSmiles]
    in_stock: list[JsonSmiles]
    solved: bool
    ai_score: float
    solvable: bool
    children: list["JsonTree"]


class Tree:
    @staticmethod
    async def from_ai(ai_tree: AiTree, f: Scorer):
        _tree = _Tree(ai_tree)
        await _tree.assign_scores_rec(f)
        return Tree(_tree)

    def __init__(self, t: Union[_Tree, JsonTree]):
        if isinstance(t, _Tree):
            assert t.expandable is not None
            assert t.in_stock is not None
            self.expandable = t.expandable
            self.in_stock = t.in_stock
            self.solved = t.solved
            self.ai_score = t.score
            self.solvable = t.solvable
            self.children = [Tree(c) for c in t.children]
        else:
            self.expandable = [Smiles.from_json(s) for s in t["expandable"]]
            self.in_stock = [Smiles.from_json(s) for s in t["in_stock"]]
            self.solved = t["solved"]
            self.ai_score = t["ai_score"]
            self.solvable = t["solvable"]
            self.children = [Tree(c) for c in t["children"]]

    def all_nodes(self) -> Iterable["Tree"]:
        return chain([self], flatten((c.all_nodes() for c in self.children)))

    def json(self) -> JsonTree:
        return {
            "expandable": [s.json() for s in self.expandable],
            "in_stock": [s.json() for s in self.in_stock],
            "solved": self.solved,
            "ai_score": self.ai_score,
            "solvable": self.solvable,
            "children": [c.json() for c in self.children],
        }

    def stats(self):
        solved = 0
        not_solved = 0
        inodes = 0
        for node in self.all_nodes():
            if node.solved:
                solved += 1
            elif node.children:
                inodes += 1
            else:
                not_solved += 1
        return {
            "solved": solved,
            "not solved": not_solved,
            "inodes": inodes,
        }
