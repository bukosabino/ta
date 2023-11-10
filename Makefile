init:
	pip install -r requirements.txt

isort:
	isort --check-only --recursive ta test

format: isort
	black --target-version py36 ta test

isort-fix:
	isort --recursive ta test

lint: isort
	prospector --no-autodetect test/
	prospector --no-autodetect ta/

test: lint
	coverage run -m unittest discover
	coverage report -m
