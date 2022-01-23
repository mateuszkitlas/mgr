run:
	PYTHONPATH="${PWD}/shared:${PYTHONPATH}" \
		~/miniconda3/envs/py3.10/bin/python \
		-m main.main

test:
	PYTHONPATH="${PWD}/shared:${PYTHONPATH}" \
		~/miniconda3/envs/py3.10/bin/python \
		-m main.test

format:
	autopep8 --in-place --recursive .
