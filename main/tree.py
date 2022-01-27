
from .score import Score
from .types import AiTree



class _CompressedTree:
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
    _CompressedTree(solved=False, solvable=False, children=[])
    ```
    """
    def __init__(self, ai: AiTree):
        self.solved = ai["is_solved"]
        self.score = ai["score"]
        self.expandable_smiles = ai["expandable_smiles"]
        self.in_stock_smiles = ai["in_stock_smiles"]
        
        true_children = (c for c in ai["children"] if c)
        compressed_children = [_CompressedTree(c) for c in true_children]
        
        self.solvable = self.solved or any(c for c in compressed_children if c.solvable)
        self.children = compressed_children if self.solvable else []

class ScoredSmiles:
    def __init__(self, smiles: str, score: Score[float]):
        self.smiles = smiles
        self.score = score

class ScoredTree:
    def __init__(
        self,
        ct: _CompressedTree,
        expandable_smiles: list[ScoredSmiles],
        in_stock_smiles: list[ScoredSmiles],
        children: list["ScoredTree"]
    ):
        self.solved = ct.solved
        self.score = ct.score
        self.expandable_smiles = expandable_smiles
        self.in_stock_smiles = in_stock_smiles
        self.solvable = ct.solvable
        self.children = children