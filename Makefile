init:
	pip install -r requirements.txt

isort:
	isort --check-only ta test

format: isort
	black --target-version py38 ta test

isort-fix:
	isort ta test

lint: isort
	prospector test/
	prospector ta/

test: lint
	coverage run -m unittest discover
	coverage report -m
