SWIG=swig
SWIGFLAGS=-python -c++

CC=g++
CPPFLAGS=-fPIC
INCLUDEPATH=/usr/include/python3.5

cpp_simple: _gaussian_model_simple.so gaussian_model_simple.py

%.py: %i
	$(SWIG) $(SWIGFLAGS) -o $@ $<

%_wrap.cpp: %.i
	$(SWIG) $(SWIGFLAGS) -o $@ $<

%.o: %.cpp
	$(CC) $(CPPFLAGS) -I $(INCLUDEPATH) -c $< -o $@

_%.so: %.o %_wrap.o
	$(CC) -shared $< -o $@


