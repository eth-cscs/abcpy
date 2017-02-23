whl_file = abcpy-0.1-py3-none-any.whl
UNITTESTS=$(shell find tests -type f -name '*_tests.py')

.DEFAULT: help
.PHONY: help test testcoverage clean doc package uninstall install reinstall

help:
	@echo Targets are: clean, test, doc, package

clean:
	find . -name "*.pyc" -type f -delete
	find . -name "__pycache__" -delete
	find . -name ".#*" -delete
	find . -name "#*#" -delete

test:
	python3 -m unittest discover -s tests -v -p "*_tests.py" || (echo "Error in unit tests."; exit 1)
	make -C doc html || (echo "Error in documentation generator."; exit 1)


testcoverage:
	command -v coverage >/dev/null 2>&1 || { echo >&2 "Python package 'coverage' has to be installed. Please, run 'pip3 install coverage'."; exit;}
	@- $(foreach TEST, $(UNITTESTS), \
		echo === Testing code coverage: $(TEST); \
		python3 -m unittest $(TEST); \
		coverage run -a --branch --source abcpy --omit \*__init__.py -m unittest $(TEST); \
	)
	coverage html -d build/testcoverage
	coverage report
	@echo
	@echo Detailed test coverage report under build/testcoverage

# documentation
doc:
	make -C doc html

# packaging
package: whl_file


uninstall:
	pip3 uninstall abcpy


install: whl_file
	pip3 install --user build/dist/$(whl_file)

reinstall: uninstall install


whl_file: clean
	python3 setup.py -v bdist_wheel -d build/dist
	@echo
	@echo "Find" `ls build/dist` "in build/dist/."
