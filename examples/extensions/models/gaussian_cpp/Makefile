SWIG=swig
SWIGFLAGS=-python -c++
WGET=wget -q

CC=g++
CPPFLAGS=-fPIC
INCLUDEPATH=$(shell python3-config --includes)
INCLUDEPATHNUMPY=$(shell python3 -c 'import numpy as np; print(np.get_include())')
PYTHONLINKERSETTINGS=$(shell python3-config --ldflags | cut -d" " -f 1)
PYTHONLIBS=$(shell python3-config --libs)

cpp_simple: _gaussian_model_simple.so gaussian_model_simple.py

clean:
	rm gaussian_model_simple.o gaussian_model_simple.py gaussian_model_simple_wrap.cpp

%.py: %.i
	$(SWIG) $(SWIGFLAGS) -o $@ $<

%_wrap.cpp: %.i numpy.i
	$(SWIG) $(SWIGFLAGS) -o $@ $<

%.o: %.cpp
	$(CC) $(CPPFLAGS) -I $(INCLUDEPATHNUMPY) $(INCLUDEPATH) -c $< -o $@

_%.so: %.o %_wrap.o
	$(CC) -shared $^ $(PYTHONLINKERSETTINGS) $(PYTHONLIBS) -o $@

%.i:
	$(WGET) "https://raw.githubusercontent.com/numpy/numpy/master/tools/swig/numpy.i"
