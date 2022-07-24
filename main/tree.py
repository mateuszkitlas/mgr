from asyncio import gather, run
from collections import defaultdict
from itertools import chain
from typing import Awaitable, Iterable, Literal, Optional, TypedDict, TypeVar, Union

from shared import Db, Fn

from .score import DictAiSmiles, Score, AiSmiles
from .types import AiTreeRaw
from .utils import flatten

T = TypeVar("T")
Scorer = Fn[tuple[str, Optional[int]], Awaitable[AiSmiles]]
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


class _AiTree:
    def __init__(self, ai: AiTreeRaw):
        solved = ai["is_solved"]
        self.ai_score = ai["ai_score"]
        self._expandable = ai["expandable"]
        self._in_stock = ai["in_stock"]
        self.children = [_AiTree(c) for c in ai["children"] if c]
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
        self.expandable: Optional[list[AiSmiles]] = None
        self.in_stock: Optional[list[AiSmiles]] = None
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

    def all_nodes(self) -> Iterable["_AiTree"]:
        return chain([self], flatten((c.all_nodes() for c in self.children)))

    async def assign_scores_rec(self, f: Scorer):
        await gather(*(t._assign_scores(f) for t in self.all_nodes()))


class AiNodeStats(TypedDict):
    count: int
    expandable: int
    in_stock: int


class AiTreeStats(TypedDict):
    internal: AiNodeStats
    solved: AiNodeStats
    not_solved: dict[int, AiNodeStats]
    max_depth: int
    max_width: int
    node_count: int
    not_solved_count: int


zero_node_stats: AiNodeStats = {
    "expandable": 0,
    "in_stock": 0,
    "count": 0,
}


def _sum_node_stats(l: list[AiNodeStats]) -> AiNodeStats:
    return {
        "expandable": sum((e["expandable"] for e in l)),
        "in_stock": sum((e["in_stock"] for e in l)),
        "count": sum((e["count"] for e in l)),
    }


def sum_tree_stats(l: list[AiTreeStats]) -> AiTreeStats:
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
        "max_depth": max(e["max_depth"] for e in l),
        "max_width": max(e["max_width"] for e in l),
        "node_count": sum(e["node_count"] for e in l),
        "not_solved_count": sum(e["not_solved_count"] for e in l),
    }


class DictAiTree(TypedDict):
    expandable: list[DictAiSmiles]
    in_stock: list[DictAiSmiles]
    ai_score: float
    children: list["DictAiTree"]
    type: TreeTypes
    not_solved_depth: int


class AiTree:
    @staticmethod
    def from_ai_dummy_scorings(ai_tree: AiTreeRaw):
        _tree = _AiTree(ai_tree)
        _tree.assign_not_solved_depth()

        async def f(x: tuple[str, Optional[int]]) -> AiSmiles:
            return AiSmiles(x[0], Score(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), x[1])

        run(_tree.assign_scores_rec(f))
        return AiTree(_tree)

    @staticmethod
    async def from_ai(ai_tree: AiTreeRaw, f: Scorer):
        _tree = _AiTree(ai_tree)
        _tree.assign_not_solved_depth()
        await _tree.assign_scores_rec(f)
        return AiTree(_tree)

    def __init__(self, data: Union[_AiTree, tuple[DictAiTree, Db]]):
        self.type: TreeTypes
        self.expandable: list[AiSmiles]
        self.in_stock: list[AiSmiles]
        self.ai_score: float
        self.children: list[AiTree]
        self.not_solved_depth: int
        if isinstance(data, _AiTree):
            assert data.expandable is not None
            assert data.in_stock is not None
            assert data.not_solved_depth is not None
            self.expandable = data.expandable
            self.in_stock = data.in_stock
            self.ai_score = data.ai_score
            self.children = [AiTree(c) for c in data.children]
            self.type = data.type
            self.not_solved_depth = data.not_solved_depth
        else:
            t, db = data
            self.expandable = [AiSmiles.from_json(s, db) for s in t["expandable"]]
            self.in_stock = [AiSmiles.from_json(s, db) for s in t["in_stock"]]
            self.ai_score = t["ai_score"]
            self.children = [AiTree((c, db)) for c in t["children"]]
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

    def all_nodes(self) -> Iterable["AiTree"]:
        return chain([self], flatten((c.all_nodes() for c in self.children)))

    def as_dict(self) -> DictAiTree:
        return {
            "expandable": [s.as_dict() for s in self.expandable],
            "in_stock": [s.as_dict() for s in self.in_stock],
            "ai_score": self.ai_score,
            "children": [c.as_dict() for c in self.children],
            "type": self.type,
            "not_solved_depth": self.not_solved_depth,
        }

    def stats(self) -> AiTreeStats:
        internal = [n for n in self.all_nodes() if n.type == "internal"]
        solved = [n for n in self.all_nodes() if n.type == "solved"]
        max_not_solved_depth = max(n.not_solved_depth for n in self.all_nodes())
        not_solved = [n for n in self.all_nodes() if n.type == "not_solved"]

        def node_stats(nodes: list[AiTree]) -> AiNodeStats:
            return {
                "count": len(nodes),
                "expandable": sum((len(n.expandable) for n in nodes)),
                "in_stock": sum((len(n.in_stock) for n in nodes)),
            }

        def max_depth(t: AiTree, acc: int) -> int:
            return max((max_depth(c, acc + 1) for c in t.children), default=acc)

        def max_width(t: AiTree):
            acc: dict[int, int] = defaultdict(lambda: 0)

            def f(u: AiTree, level: int):
                acc[level] += len(u.children)
                for w in u.children:
                    f(w, level + 1)

            f(t, 0)
            return max(acc.values())

        def node_count(t: AiTree) -> int:
            return 1 + sum(node_count(c) for c in t.children)

        return {
            "internal": node_stats(internal),
            "solved": node_stats(solved),
            "not_solved": {
                depth: node_stats(
                    [n for n in not_solved if n.not_solved_depth == depth]
                )
                for depth in range(max_not_solved_depth + 1)
            },
            "max_depth": max_depth(self, 0),
            "max_width": max_width(self),
            "node_count": node_count(self),
            "not_solved_count": len([n for n in not_solved if n.not_solved_depth == 0]),
        }
