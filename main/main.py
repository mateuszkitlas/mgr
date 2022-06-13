from asyncio import gather, run
from sys import argv
from shared import CondaApp

from main.tree import Tree

from .data import Db, data
from .graph import Graph
from .helpers import Scoring, all_scorings, app_ai_ac, app_scorers
from .types import AiInput, AiTree, Setup, Timed

uw_multipliers = [0.15, 0.4, 0.6]


def n(first: float, second: float):
    return (first, second, False)


setups: list[Setup] = [
    {"score": "sc", "uw_multiplier": 0.0, "normalize": n(2.5, 4.5), "agg": "max",},
    *[
        {
            "score": "sc",
            "uw_multiplier": uw_multiplier,
            "normalize": normalize,
            "agg": "max",
        }
        for uw_multiplier in uw_multipliers
        for normalize in [n(2.5, 4.5), n(3.0, 4.5), n(3.5, 4.5)]
    ],
    *[
        {
            "score": "sa",
            "uw_multiplier": uw_multiplier,
            "normalize": normalize,
            "agg": "max",
        }
        for uw_multiplier in uw_multipliers
        for normalize in [n(3.0, 6.0), n(4.0, 6.0), n(5.0, 6.0)]
    ],
    *[
        {
            "score": "mf",
            "uw_multiplier": uw_multiplier,
            "normalize": (-700.0, 0.0, True),
            "agg": "min",
        }
        for uw_multiplier in uw_multipliers
    ],
]


async def ai_ac():
    mols = data()
    with Db() as db:
        async with app_ai_ac() as (ai, ac):
            for i, mol in enumerate(mols):
                print(f"{i}/{len(mols)}")
                # for i, setup in enumerate(setups):
                await gather(
                    db.maybe_create(["ai", mol.smiles], lambda: ai(mol.smiles)),
                    db.maybe_create(["ac", mol.smiles], lambda: ac(mol.smiles)),
                )


async def ai():
    mols = data()
    with Db() as db:
        async with CondaApp[AiInput, Timed[AiTree]](4001, "ai", "aizynth-env") as (fetch, _):
            for j, mol in enumerate(mols):
                for i, setup in enumerate(setups):
                    ai_input: AiInput = {"smiles": mol.smiles, "setup": setup}
                    print(f"[setup {i}/{len(setups)}][mol {j}/{len(mols)}]")
                    await db.maybe_create(["ai", ai_input], lambda: fetch(ai_input))
                

async def score_ac_smileses():
    mols = data()
    with Db() as db:
        with_nones = (db.read(["ac", mol.smiles]) for mol in mols)
        graphs = (Graph.from_ac_result(data) for data in with_nones if data)
        smileses = [smiles for g in graphs for smiles in g.all_smileses()]
        all = len(smileses)
        n = 0
        async with app_scorers() as (scorer, smileser):

            async def f(raw_smiles: str):
                smiles = await db.read_or_create(
                    ["smiles", raw_smiles], lambda: smileser(raw_smiles)
                )

                async def g(scoring: Scoring):
                    score = await db.read_or_create(
                        [scoring, raw_smiles], lambda: scorer(scoring, raw_smiles)
                    )
                    db.write([scoring, smiles], score)

                await gather(*[g(scoring) for scoring in all_scorings])
                nonlocal n
                n += 1
                print(f"{n}/{all}")

            await gather(*[f(smiles) for smiles in smileses])


def ac_graph_to_tree():
    mols = data()
    with Db() as db:
        with_nones = (db.read(["ac", mol.smiles]) for mol in mols)
        graphs = (Graph.from_ac_result(data) for data in with_nones)
        for graph in graphs:
            print(Tree.from_ac(graph, db).stats())


if __name__ == "__main__":
    if argv[1] == "ai_ac":
        run(ai_ac())
    elif argv[1] == "ai":
        run(ai())
    elif argv[1] == "score_ac_smileses":
        run(score_ac_smileses())
    elif argv[1] == "ac_graph_to_tree":
        ac_graph_to_tree()


"""
def ac_tag():
    mols = data()

    def f(a: int, b: int, c: int):
        s = float(a + b + c) / 100
        if s > 0.0:
            return f"{int(a/s)}%\t{int(b/s)}%\t{int(c/s)}%"
        else:
            return f"-\t-\t-"

    with Db() as db:
        raws = (db.read(["ac", mol.smiles]) for mol in mols)
        all = [r for r in raws if r]
        tags = [tag(r) for r in all]
        tags_pretty = sorted(
            [
                (solved, (c["not_solved"], c["internal"], c["solved"]))
                for _smiles, solved, c in tags
            ]
        )
        solved = sum(solved for _smiles, solved, _counts in tags)
        print(
            (
                "solved?",
                "not_solved chemicals",
                "internal chemicals",
                "solved chemicals",
            )
        )
        for ok, counts in tags_pretty:
            ok2 = "solved!" if ok else "not solved :("
            print(f"{ok2}\t{f(*counts)}\t{counts}")
        print(f"{solved}/{len(all)}")


if __name__ == "__main__":
    if argv[1] == "ai_ac":
        run(ai_ac())
    elif argv[1] == "ac_tag":
        ac_tag()
    trees = load_trees(json_file)
    done = {_hash(ai_input) for ai_input, _ in trees}
    async with app_ai() as ai:  # , app_scorers() as scorer:
        for j, mol in enumerate(mols):
            for i, setup in enumerate(setups):
                ai_input: AiInput = {"smiles": mol.smiles, "setup": setup}
                txt = f"[setup {i}/{len(setups)}][mol {j}/{len(mols)}]"
                if _hash(ai_input) in done:
                    print(f"{txt}[skipped]")
                else:
                    print(f"{txt} ", end="", flush=True)
                    ai_tree = await ai.tree(setup, mol.smiles)
                    # print(f"][from_ai...", end = "", flush=True)
                    # real_time, tree = await Timer.acalc(
                    #     Tree.from_ai(ai_tree, scorer.score)
                    # )
                    # scorer.add_real_time(real_time)
                    # trees.append((ai_input, tree.json()))
                    trees.append((ai_input, ai_tree))
                    save_trees(trees, json_file)


def dummy_scorings():
    trees = load_trees(json_file)
    per_mol: dict[str, list[Tuple[Setup, Tree]]] = {
        ai_input["smiles"]: [] for ai_input, _ in trees
    }
    for ai_input, ai_tree in trees:
        tree = Tree.from_ai_dummy_scorings(ai_tree)
        per_mol[ai_input["smiles"]].append((ai_input["setup"], tree))
    for smiles, lst in per_mol.items():
        print(smiles)
        for setup, tree in lst:
            stats = tree.stats()
            prefix = (
                "zero_setup"
                if setup["uw_multiplier"] == 0.0
                else f"{setup['score']}, n{setup['normalize'][0]}, uw{setup['uw_multiplier']}"
            )
            print(
                f"{prefix} ",
                {
                    "internal": stats["internal"]["count"],
                    "solved": stats["solved"]["count"],
                    "not_solved": sum(
                        [s["count"] for s in stats["not_solved"].values()]
                    ),
                },
            )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "dummy_scorings":
        dummy_scorings()
    elif len(sys.argv) == 1:
        run(main())
    else:
        print("invalid sys.argv")
"""
