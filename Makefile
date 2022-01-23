run:
	${CONDA_PREFIX}/envs/py3.10/bin/python -m main.main

test:
	${CONDA_PREFIX}/envs/py3.10/bin/python -m main.test

format:
	${CONDA_PREFIX}/envs/py3.10/bin/black */*.py
