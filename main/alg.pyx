from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple, TypeVar
from aizynthfinder.aizynthfinder import AiZynthFinder, AiZynthExpander
import os

paracetamol = "CC(=O)Nc1ccc(O)cc1"
home = os.environ["HOME"]

class AiZynth:
  def __init__(self):
    self.config_path = f"{home}/aizynthfinder/aizynthfinder.config.yml"

    self.finder = AiZynthFinder(self.config_path)
    self.finder.stock.select("zinc")
    self.finder.expansion_policy.select("full_uspto")
    #self.finder.filter_policy.select("full_uspto")

    #self.expander = AiZynthExpander(self.config_path)
    #self.expander.expansion_policy.select("full_uspto")
    #self.expander.filter_policy.select("uspto")

  def _find(self, smiles: str):
    self.finder.target_smiles = smiles
    self.finder.tree_search()
    self.finder.build_routes()
    self.finder.routes.compute_scores(*self.finder.scorers.objects())
    
  def tree(self, smiles: str):
    self._find(smiles)
    ai_tree: Dict[str, Any] = self.finder.tree.serialize2() # type: ignore
    _, tree = convert(ai_tree)
    return tree
  
  def find(self, smiles: str):
    self._find(smiles)
    return [x["all_score"] for x in self.finder.routes if x["node"].state.is_solved]



from client import fetch

def ra_score(smiles: str) -> float:
  return fetch("localhost", 4000, smiles)

MinMaxAvg = Tuple[float, float, float]
def avg(vals: List[float]) -> float: return sum(vals)/len(vals)
def min_max_avg(vals: List[float]) -> MinMaxAvg:
  return (min(vals), max(vals), avg(vals))

class Scores:
  def __init__(self, ai: float, ra: MinMaxAvg, smileses: List[str]):
    self.ai = ai #aizynth node score
    self.ra = ra
    self.smilses 
# trzymać smajlsy


# Solved
# Node - rename to solvable
# NotSolved - rename to not solvable

Node
  - Solved
  - Solved
  - Solved
# zachowywać statystyki czasu liczenia

# --- trzymać wszystkie smilesy
# 1. Node taki ze: ma syns Node i NotSolved
"""
ra - avg
for (not_solved: float, node: float)
  
  
List[Tuple[float, float]]
Scatter plot
"""

# 2. analogicznie jak w 1.
"""
ra - avg
for (not_solved: float, node: float)
  not_solved - float
  
List[float] - histogram z jakąś granulacją
"""

# 3. policzyć Scores dla Solved (bez aizynth scora). bierzemy pary Solved i Node. analogicznie doo 1. i 2.
# 4. (wszystkie NotSolved) i (wszystkie Node lub Solved) - boxplot
#  - ad. (wszystkie NotSolved) i (wszystkie Node) - boxplot score ai

class AggScores:
  def __init__(self, ai: MinMaxAvg, ra: MinMaxAvg):
    self.ai = ai
    self.ra = ra
    self.smiles_with_scores: List[Tuple[str, Scores]]
  @staticmethod
  def _agg(l: List[Scores]):
    if l:
      return AggScores(
        min_max_avg([s.ai for s in l]),
        (
          min([s.ra[0] for s in l]),
          max([s.ra[1] for s in l]),
          avg([s.ra[2] for s in l]),
        )
      )
  @staticmethod
  def agg(l: List["HasSmileses"]):
    return AggScores._agg([c.scores for c in l])


class AggScoresBoth:
  def __init__(self, solvable: Optional[AggScores], not_solvable: Optional[AggScores]):
    self.solvable = solvable
    self.not_solvable = not_solvable




class T:
  def json(self) -> Dict[str, Any]: raise NotImplementedError
  def solvable(self) -> bool: raise NotImplementedError
  def solved(self) -> bool: raise NotImplementedError
  children: List["T"]

class HasSmileses(T):
  smileses: List[str]
  score: float
  def solved(self): return False
  @cached_property
  def scores(self):
    return Scores(
      ai = self.score,
      ra = min_max_avg([ra_score(s) for s in self.smileses]),
    )

class Solved(T):
  children: List[T] = []
  def solved(self): return True
  def solvable(self): return True
  def json(self):
    return {'type': "Solved"}

class NotSolved(HasSmileses):
  children: List[T] = []
  def solvable(self): return False
  def json(self):
    return {'type': "NotSolved", "smileses": self.smileses, "score": self.score}
  def __init__(self, smileses: List[str], score: float):
    self.smileses = smileses
    self.score = score

X = TypeVar("X")
def flatten(l: List[List[X]]) -> List[X]:
  return [item for sublist in l for item in sublist]

class Node(HasSmileses):
  def solvable(self): return True
  def json(self):
    return {
      'type': "Node",
      'children': [n.json() for n in self.children],
      'smileses': self.smileses,
      'score': self.score,
    }
  def __init__(self, children: List[T], smileses: List[str], score: float):
    self.children = children
    self.smileses = smileses
    self.score = score

  @cached_property
  def children_not_solved(self):
    return [c for c in self.children if isinstance(c, NotSolved)]

  @cached_property
  def children_node(self):
    return [c for c in self.children if isinstance(c, Node)]

  @cached_property
  def agg_scores_both(self):
    return AggScoresBoth(
      AggScores.agg(self.children_node),
      AggScores.agg(self.children_not_solved),
    )
  def all_nodes(self) -> List["Node"]:
    return [self] + flatten([n.all_nodes() for n in self.children_node])

def convert(ai_tree: Dict[str, Any]) -> Tuple[bool, T]:
  children: List[Dict[str, Any]] = [t for t in ai_tree['children'] if t]
  solved: bool = ai_tree['is_solved']
  smileses: List[str] = ai_tree['expandable_smileses']
  score: float = ai_tree['score']
  if solved:
    return (True, Solved())
  elif children == []:
    return (False, NotSolved(smileses, score))
  else:
    tmp: Tuple[List[bool], List[T]] = zip(*(convert(t) for t in children)) # type: ignore
    paths, nodes = tmp
    if any(paths):
      return (True, Node(nodes, smileses, score))
    else:
      return (False, NotSolved(smileses, score))



class AggTree:
  def __init__(self, node: Node):
    AggScoresBoth(



