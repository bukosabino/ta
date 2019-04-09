init:
	pip install -r dev-requirements.txt

lint:
	flake8 ta && isort --check-only --recursive ta

isort-fix:
	isort --recursive ta
