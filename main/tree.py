from asyncio import gather, run
from itertools import chain
from random import random
from typing import (Awaitable, Iterable, Literal, Optional, TypedDict, TypeVar,
                    Union)

from main.data import Db
from main.graph import Graph
from shared import Fn

from .score import JsonSmiles, Score, Smiles
from .types import AiTree
from .utils import flatten, not_none

T = TypeVar("T")
Scorer = Fn[tuple[str, Optional[int]], Awaitable[Smiles]]
TreeTypes = Literal["solved", "not_solved", "internal"]


class Mols:
    def __init__(self, mols_and_expands_or_root: Union[tuple["Mols", str], str]):
        self._expandable: set[str]
        self._in_stock: set[str]
        self._expanded: set[str]
        self.expands: Optional[str]
        if isinstance(mols_and_expands_or_root, tuple):
            mols, expands = mols_and_expands_or_root
            self._expandable = mols._expandable.copy()
            self._in_stock = mols._in_stock.copy()
            self._expanded = mols._expanded.copy()
            self.expands = expands
        else:
            self._expandable = set([mols_and_expands_or_root])
            self._in_stock = set()
            self._expanded = set()
            self.expands = None

    def dump(self, db: Db):
        def f(s: set[str]):
            # TODO transforms
            return [Smiles.from_db(smiles, None, db) for smiles in s]

        return f(self._expandable), f(self._in_stock)

    def _expand(self, expandable_smiles: str, smarts: str, graph: Graph):
        substrates = [
            substrate
            for substrate in graph.substrates(smarts)
            if substrate not in self._expanded
            and substrate not in self._in_stock
            and substrate not in self._expandable
        ]
        mols = Mols((self, expandable_smiles))
        mols._expanded.add(expandable_smiles)
        if substrates:
            mols._expandable.remove(expandable_smiles)
            mols._in_stock.add(expandable_smiles)
            for substrate, buyable in substrates:
                if buyable:
                    mols._in_stock.add(substrate)
                else:
                    mols._expandable.add(substrate)
        return mols

    def expand_all(self, graph: Graph):
        return not_none(
            self._expand(expandable_smiles, smarts, graph)
            for expandable_smiles in (self._expandable - self._expanded)
            for smarts in graph.out_nodes(expandable_smiles)
        )


TreeFromGraph = tuple[Mols, list["TreeFromGraph"]]


class _Tree:
    def __init__(self, ai: AiTree):
        solved = ai["is_solved"]
        self.ai_score = ai["ai_score"]
        self._expandable = ai["expandable"]
        self._in_stock = ai["in_stock"]
        self.children = [_Tree(c) for c in ai["children"] if c]
        if solved:
            assert not self.children
        solvable = solved or any(
            c for c in self.children if c.type in ("solved", "internal")
        )
        self.type: TreeTypes = (
            "solved" if solved else ("internal" if solvable else "not_solved")
        )
        assert (
            (self.type == "solved" and (not self._expandable) and self._in_stock)
            or (self.type == "not_solved" and self._expandable)
            or (self.type == "internal" and self._expandable)
        )
        self.expandable: Optional[list[Smiles]] = None
        self.in_stock: Optional[list[Smiles]] = None
        self.not_solved_depth: Optional[int] = None

    def _assign_not_solved_depth(self, ancestor_not_solved_depth: int):
        if self.type in ["solved", "internal"]:
            self.not_solved_depth = -1
        else:
            self.not_solved_depth = ancestor_not_solved_depth + 1
        for c in self.children:
            c._assign_not_solved_depth(self.not_solved_depth)

    def assign_not_solved_depth(self):
        self._assign_not_solved_depth(-1)

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
    expandable: int
    in_stock: int


class TreeStats(TypedDict):
    internal: NodeStats
    solved: NodeStats
    not_solved: dict[int, NodeStats]


zero_node_stats: NodeStats = {
    "expandable": 0,
    "in_stock": 0,
    "count": 0,
}


def _sum_node_stats(l: list[NodeStats]) -> NodeStats:
    return {
        "expandable": sum((e["expandable"] for e in l)),
        "in_stock": sum((e["in_stock"] for e in l)),
        "count": sum((e["count"] for e in l)),
    }


def sum_tree_stats(l: list[TreeStats]) -> TreeStats:
    max_depth = max(flatten(e["not_solved"].keys() for e in l), default=-1)
    return {
        "internal": _sum_node_stats([e["internal"] for e in l]),
        "solved": _sum_node_stats([e["solved"] for e in l]),
        "not_solved": {
            depth: _sum_node_stats(
                [e["not_solved"].get(depth, zero_node_stats) for e in l]
            )
            for depth in range(max_depth + 1)
        },
    }


class JsonTree(TypedDict):
    expandable: list[JsonSmiles]
    in_stock: list[JsonSmiles]
    ai_score: float
    children: list["JsonTree"]
    type: TreeTypes
    not_solved_depth: int


class Tree:
    @staticmethod
    def from_ai_dummy_scorings(ai_tree: AiTree):
        _tree = _Tree(ai_tree)
        _tree.assign_not_solved_depth()

        async def f(x: tuple[str, Optional[int]]) -> Smiles:
            return Smiles(x[0], Score(0.0, 0.0, 0.0, 0.0, 0.0), x[1])

        run(_tree.assign_scores_rec(f))
        return Tree(_tree)

    @staticmethod
    async def from_ai(ai_tree: AiTree, f: Scorer):
        _tree = _Tree(ai_tree)
        _tree.assign_not_solved_depth()
        await _tree.assign_scores_rec(f)
        return Tree(_tree)

    @staticmethod
    def from_ac(graph: Graph, db: Db):
        def f(
            mols: Mols,
        ) -> TreeFromGraph:
            return (
                mols,
                [f(child_mols) for child_mols in mols.expand_all(graph)],
            )

        t = Tree((f(Mols(graph.root)), db))
        if t.type == "not_solved":
            if graph.ac_paths_count:
                print(
                    f"invalid tree from graph. there should be at least {graph.ac_paths_count} paths"
                )
        return t

    def __init__(self, t: Union[_Tree, JsonTree, tuple[TreeFromGraph, Db]]):
        self.type: TreeTypes
        self.expandable: list[Smiles]
        self.in_stock: list[Smiles]
        self.ai_score: float
        self.children: list[Tree]
        self.not_solved_depth: int
        if isinstance(t, _Tree):
            assert t.expandable is not None
            assert t.in_stock is not None
            assert t.not_solved_depth is not None
            self.expandable = t.expandable
            self.in_stock = t.in_stock
            self.ai_score = t.ai_score
            self.children = [Tree(c) for c in t.children]
            self.type = t.type
            self.not_solved_depth = t.not_solved_depth
        elif isinstance(t, tuple):
            (mols, tfgs), db = t
            self.expandable, self.in_stock = mols.dump(db)
            self.ai_score = random()  # TODO
            self.children = [Tree((tfg, db)) for tfg in tfgs]
            self.type = (
                (
                    "internal"
                    if any(
                        tree.type == "solved" or tree.type == "internal"
                        for tree in self.children
                    )
                    else "not_solved"
                )
                if self.expandable
                else "solved"
            )
            self.not_solved_depth = 0  # TODO
        else:
            self.expandable = [Smiles.from_json(s) for s in t["expandable"]]
            self.in_stock = [Smiles.from_json(s) for s in t["in_stock"]]
            self.ai_score = t["ai_score"]
            self.children = [Tree(c) for c in t["children"]]
            self.type = t["type"]
            self.not_solved_depth = t["not_solved_depth"]
        assert (
            (
                self.type == "solved"
                and (not self.expandable)
                and self.in_stock
                and (not self.children)
            )
            or (self.type == "not_solved" and self.expandable)
            or (self.type == "internal" and self.expandable)
        ), (self.type, len(self.expandable), len(self.in_stock), len(self.children))
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
            "not_solved_depth": self.not_solved_depth,
        }

    def stats(self) -> TreeStats:
        internal = [n for n in self.all_nodes() if n.type == "internal"]
        solved = [n for n in self.all_nodes() if n.type == "solved"]
        max_depth = max(n.not_solved_depth for n in self.all_nodes())
        not_solved = [n for n in self.all_nodes() if n.type == "not_solved"]

        def node_stats(nodes: list[Tree]) -> NodeStats:
            return {
                "count": len(nodes),
                "expandable": sum((len(n.expandable) for n in nodes)),
                "in_stock": sum((len(n.in_stock) for n in nodes)),
            }

        return {
            "internal": node_stats(internal),
            "solved": node_stats(solved),
            "not_solved": {
                depth: node_stats(
                    [n for n in not_solved if n.not_solved_depth == depth]
                )
                for depth in range(max_depth + 1)
            },
        }
