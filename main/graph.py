from queue import Queue
from typing import Any, Iterable, Tuple, Union, cast

from networkx.readwrite import json_graph

from main.types import AcGraphChemical, AcGraphReaction, AcResult


class NetworkXError(Exception):
    pass


MAX_DEPTH = 6


class Graph:
    def __init__(self, raw: Any, root: str, ac_paths_count: int):
        self._range: set[str] = set()
        self.root = root
        self.ac_paths_count = ac_paths_count
        try:
            self._g: Any = None if raw == {} else json_graph.node_link_graph(raw)
        except KeyError as e:
            raise NetworkXError from e
        if self._g:
            q: Queue[tuple[str, int]] = Queue()
            q.put((self.root, 0))
            while not q.empty():
                smiles, depth = q.get()
                if smiles not in self._range:
                    self._range.add(smiles)
                    if depth < MAX_DEPTH * 2:
                        for s in self._g.successors(smiles):
                            q.put((s, depth + 1))
        else:
            self._range.add(self.root)

    def chemical_buyable(self, smiles: str):
        if self._g:
            v: AcGraphChemical = self._g.nodes[smiles]
            assert v["type"] == "chemical"
            if v["terminal"]:
                assert any(True for _ in self.out_nodes(smiles)) == False
                return v["purchase_price"] >= 0.0
            else:
                return False
        else:
            assert smiles == self.root
            return False

    def _successors(self, smiles: str) -> Iterable[str]:
        return self._g.successors(smiles)

    def out_nodes(self, smiles: str):
        return (
            (s for s in self._successors(smiles) if s in self._range)
            if self._g
            else cast(list[str], [])
        )

    def substrates(self, smarts: str):
        return (
            (smiles, self.chemical_buyable(smiles)) for smiles in self.out_nodes(smarts)
        )

    def in_nodes(self, smiles: str) -> Iterable[str]:
        return self._g.predecessors(smiles)

    def all_smileses(self):
        if self._g:
            it: Iterable[
                Tuple[str, Union[AcGraphChemical, AcGraphReaction]]
            ] = self._g.nodes.data()
            return (smiles for (smiles, v) in it if v["type"] == "chemical")
        else:
            return [self.root]

    @classmethod
    def from_ac_result(cls, ac_result: AcResult):
        [_time, [smiles, paths, raw]] = ac_result
        res = cls(raw, smiles, len(paths))
        # assert res.chemical_solved(res.root) == bool(paths), [paths, raw]
        return res
