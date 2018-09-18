F2PY=python -m numpy.f2py
EXT_SUFFIX := $(shell python3-config --extension-suffix)

default: gaussian_model_simple$(EXT_SUFFIX)

%$(EXT_SUFFIX): %.f90
	echo $(F2PY)
	$(F2PY) -c -m $* $<

