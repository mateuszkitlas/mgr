# Install conda environments

```sh
# aizynth-env
conda env create -f https://raw.githubusercontent.com/MolecularAI/aizynthfinder/master/env-users.yml

# scorers
conda env create -f conda_scorers.yml

# py3.10 for main app
conda env create -f conda_py3.10.yml
```

# Computation time (in seconds)
## Smiles in main/test.csv
```
Avarage timings {"sa": 0.007827979591836733, "sc": 0.4385612653061225, "ra": 4.057286632653062, "syba": 0.009300346938775511}
Total time: 221.13583500000001, Real time: 15.883786
```
## Loading scorers
```json
{"ra": 2.316089, "sa": 0.000566, "sc": 0.043805, "syba": 120.992621}
```