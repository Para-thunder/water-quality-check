install:
	pip install -r requirements.txt

test:
	pytest

lint:
	flake8 src/

run:
	python src/main.py
