init:
	pip install -r requirements.txt

lint:
	pycodestyle --max-line-length=119 ta && isort --check-only --recursive ta

isort-fix:
	isort --recursive ta

test: lint
	# python -m unittest discover
	coverage run tests.py
	coverage report -m
