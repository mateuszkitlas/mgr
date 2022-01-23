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
## Smiles
```json
{sa: 0.012774, sc: 1.48846, ra: 13.26315, syba: 0.00075}
```
## Loading scorers
```json
{ra: 0.00052, sa: 0.00035, sc: 0.21158, syba: 133.66499}
```