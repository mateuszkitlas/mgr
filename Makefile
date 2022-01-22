run:
	PYTHONPATH="${PWD}/shared:${PYTHONPATH}" \
		~/miniconda3/envs/py3.10/bin/python \
		-m main.main

format:
	autopep8 --in-place --recursive .
