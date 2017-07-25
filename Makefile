VERSION=$(shell cat VERSION)
UNITTESTS=$(shell find tests -type f -name '*_tests.py')
MAKEDIRS=$(shell find examples -name Makefile -exec dirname {} \;)
whl_file = abcpy-${VERSION}-py3-none-any.whl

.DEFAULT: help
.PHONY: help clean doc doctest exampletest package test uninstall unittest install reinstall $(MAKEDIRS)

help:
	@echo Targets are: clean, doc, doctest, exampletest, package, uninstall, unittest, test

clean:
	find . -name "*.pyc" -type f -delete
	find . -name "__pycache__" -delete
	find . -name ".#*" -delete
	find . -name "#*#" -delete

test: unittest unittest_mpi exampletest doctest

unittest:
	python3 -m unittest discover -s tests -v -p "*_tests.py" || (echo "Error in unit tests."; exit 1)


unittest_mpi:
	mpirun -np 4 python3 -m unittest discover -s tests -v -p "backend_mpi_tests.py" || (echo "Error in unit tests."; exit 1)


$(MAKEDIRS):
	make -C $@

doctest:
	make -C doc html || (echo "Error in documentation generator."; exit 1)

exampletest: $(MAKEDIRS)
	python3 -m unittest discover -s examples -v -p "*.py" || (echo "Error in example tests."; exit 1)

coveragetest:
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
