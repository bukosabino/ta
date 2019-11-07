init:
	pip install -r dev-requirements.txt

lint:
	pycodestyle --max-line-length=119 ta && isort --check-only --recursive ta

isort-fix:
	isort --recursive ta

test:
	python -m unittest discover
