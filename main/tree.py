from asyncio import gather
from itertools import chain
from typing import (Awaitable, Iterable, Literal, Optional, Tuple, TypedDict,
                    TypeVar, Union)

from shared import Fn

from .score import JsonSmiles, Smiles
from .types import AiTree
from .utils import flatten

T = TypeVar("T")
Scorer = Fn[Tuple[str, Optional[int]], Awaitable[Smiles]]
TreeTypes = Literal["solved", "not_solved", "internal"]


class _Tree:
    """
    Turns nodes like this:
    ```py
    AiTree(is_solved=False, children=[
        AiTree(is_solved=False),
        AiTree(is_solved=False),
        AiTree(is_solved=False),
    ])
    ```
    To one node:
    ```py
    _Tree(type="not_solved", children=[])
    ```
    """

    def __init__(self, ai: AiTree):
        solved = ai["is_solved"]
        self.ai_score = ai["ai_score"]
        self._expandable = ai["expandable"]
        self._in_stock = ai["in_stock"]
        true_children = [c for c in ai["children"] if c]
        if solved:
            assert not true_children
        compressed_children = [_Tree(c) for c in true_children]
        solvable = solved or any(
            c for c in compressed_children if c.type in ("solved", "internal")
        )
        self.children = compressed_children if solvable else []
        self.type: TreeTypes = "solved" if solved else (
            "internal" if solvable else "not_solved"
        )
        assert (
            (self.type == "solved" and (not self._expandable) and self._in_stock)
            or (self.type == "not_solved" and self._expandable)
            or (self.type == "internal" and self._expandable)
        )
        self.expandable: Optional[list[Smiles]] = None
        self.in_stock: Optional[list[Smiles]] = None

    async def _assign_scores(self, f: Scorer):
        assert self.expandable is None
        assert self.in_stock is None
        expandable, in_stock = await gather(
            gather(*(f(e) for e in self._expandable)),
            gather(*(f(e) for e in self._in_stock)),
        )
        self.expandable, self.in_stock = list(expandable), list(in_stock)

    def all_nodes(self) -> Iterable["_Tree"]:
        return chain([self], flatten((c.all_nodes() for c in self.children)))

    async def assign_scores_rec(self, f: Scorer):
        await gather(*(t._assign_scores(f) for t in self.all_nodes()))


class NodeStats(TypedDict):
    count: int
    expandable_smiles: int
    in_stock_smiles: int


TreeStats = dict[TreeTypes, NodeStats]


def sum_node_stats(l: list[NodeStats]) -> NodeStats:
    return {
        "expandable_smiles": sum((e["expandable_smiles"] for e in l)),
        "in_stock_smiles": sum((e["in_stock_smiles"] for e in l)),
        "count": sum((e["count"] for e in l)),
    }


def sum_tree_stats(l: list[TreeStats]) -> TreeStats:
    return {
        "internal": sum_node_stats([e["internal"] for e in l]),
        "solved": sum_node_stats([e["solved"] for e in l]),
        "not_solved": sum_node_stats([e["not_solved"] for e in l]),
    }


class JsonTree(TypedDict):
    expandable: list[JsonSmiles]
    in_stock: list[JsonSmiles]
    ai_score: float
    children: list["JsonTree"]
    type: TreeTypes


class Tree:
    @staticmethod
    async def from_ai(ai_tree: AiTree, f: Scorer):
        _tree = _Tree(ai_tree)
        await _tree.assign_scores_rec(f)
        return Tree(_tree)

    def __init__(self, t: Union[_Tree, JsonTree]):
        self.type: TreeTypes
        if isinstance(t, _Tree):
            assert t.expandable is not None
            assert t.in_stock is not None
            self.expandable = t.expandable
            self.in_stock = t.in_stock
            self.ai_score = t.ai_score
            self.children = [Tree(c) for c in t.children]
            self.type = t.type
        else:
            self.expandable = [Smiles.from_json(s) for s in t["expandable"]]
            self.in_stock = [Smiles.from_json(s) for s in t["in_stock"]]
            self.ai_score = t["ai_score"]
            self.children = [Tree(c) for c in t["children"]]
            self.type = t["type"]
        assert (
            (self.type == "solved" and (not self.expandable) and self.in_stock)
            or (self.type == "not_solved" and self.expandable)
            or (self.type == "internal" and self.expandable)
        )
        assert self.expandable or self.in_stock

    def all_nodes(self) -> Iterable["Tree"]:
        return chain([self], flatten((c.all_nodes() for c in self.children)))

    def json(self) -> JsonTree:
        return {
            "expandable": [s.json() for s in self.expandable],
            "in_stock": [s.json() for s in self.in_stock],
            "ai_score": self.ai_score,
            "children": [c.json() for c in self.children],
            "type": self.type,
        }

    def stats(self) -> TreeStats:
        internal = [n for n in self.all_nodes() if n.type == "internal"]
        solved = [n for n in self.all_nodes() if n.type == "solved"]
        not_solved = [n for n in self.all_nodes() if n.type == "not_solved"]

        def node_stats(nodes: list[Tree]) -> NodeStats:
            return {
                "count": len(nodes),
                "expandable_smiles": sum((len(n.expandable) for n in nodes)),
                "in_stock_smiles": sum((len(n.in_stock) for n in nodes)),
            }

        return {
            "internal": node_stats(internal),
            "solved": node_stats(solved),
            "not_solved": node_stats(not_solved),
        }
