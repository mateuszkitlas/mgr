# Install conda environments

```sh
# aizynth-env
conda env create -f https://raw.githubusercontent.com/MolecularAI/aizynthfinder/master/env-users.yml

# scorers
conda env create -f conda_scorers.yml

# py3.10 for main
conda env create -f conda_py3.10.yml

```

# Computation time (in seconds)
## Smiles
```json
{'sa': 0.012774142857142854, 'sc': 1.4884573877551022, 'ra': 13.263153061224488, 'syba': 0.0007510612244897959}
```
## Loading scorers
```json
{'ra': 0.000519, 'sa': 0.000346, 'sc': 0.211584, 'syba': 133.664985}
```