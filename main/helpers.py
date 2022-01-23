from .conda_app import CondaApp
from .types import RawScore, Timed, AiTree

app_ai = CondaApp[str, Timed[AiTree]](4000, "ai", "aizynth-env")
app_scorers = CondaApp[str, RawScore](4001, "scorers", "scorers")
