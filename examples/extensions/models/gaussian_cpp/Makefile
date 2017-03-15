SWIG=swig
SWIGFLAGS=-python -c++
WGET=wget -q

CC=g++
CPPFLAGS=-fPIC
INCLUDEPATH=/usr/include/python3.4m

cpp_simple: _gaussian_model_simple.so gaussian_model_simple.py

%.py: %.i
	$(SWIG) $(SWIGFLAGS) -o $@ $<

%_wrap.cpp: %.i numpy.i
	$(SWIG) $(SWIGFLAGS) -o $@ $<

%.o: %.cpp
	$(CC) $(CPPFLAGS) -I $(INCLUDEPATH) -c $< -o $@

_%.so: %.o %_wrap.o
	$(CC) -shared $< -o $@

%.i:
	$(WGET) "https://raw.githubusercontent.com/numpy/numpy/master/tools/swig/numpy.i"
