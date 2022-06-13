from typing import Any, Literal, Optional, Tuple, TypedDict, TypeVar

T = TypeVar("T")
Timed = Tuple[float, T]


class AiTree(TypedDict):
    is_solved: bool
    ai_score: float
    expandable: list[Tuple[str, Optional[int]]]
    in_stock: list[Tuple[str, Optional[int]]]
    children: list[Optional["AiTree"]]


class _AcPathChemical_Attributes(TypedDict):
    depth: int


class AcPathChemical(TypedDict):
    attributes: _AcPathChemical_Attributes
    smiles: str
    ppg: float
    as_reactant: int
    as_product: int
    terminal: bool
    is_chemical: Literal[True]
    id: str
    children: list["AcPathReaction"]


class AcPathReaction(TypedDict):
    smiles: str
    tforms: list[str]
    template_score: float
    plausibility: float
    is_reaction: Literal[True]
    rank: int
    num_examples: int
    necessary_reagent: str
    precursor_smiles: str
    rms_molwt: float
    num_rings: int
    scscore: float
    id: str
    children: list["AcPathChemical"]


AcResult = Timed[Tuple[str, list[AcPathChemical], Any]]


class AcGraphChemical(TypedDict):
    as_product: int
    as_reactant: int
    purchase_price: float
    terminal: bool
    type: Literal["chemical"]


class AcGraphReaction(TypedDict):
    necessary_reagent: str
    num_examples: int
    num_rings: int
    plausibility: float
    precursor_smiles: str
    rank: int
    rms_molwt: float
    scscore: float
    template_score: float
    tforms: list[str]
    type: Literal["reaction"]


class Setup(TypedDict):
    agg: Literal["max", "min"]
    score: Literal["sa", "sc", "mf"]
    uw_multiplier: float
    normalize: Tuple[float, float, bool]


class AiInput(TypedDict):
    smiles: str
    setup: Setup
