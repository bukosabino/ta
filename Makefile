init:
	pip install -r dev-requirements.txt

lint:
	# flake8 ta && isort --check-only --recursive ta
	pycodestyle --max-line-length=119 ta && isort --check-only --recursive ta

isort-fix:
	isort --recursive ta
