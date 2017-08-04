F2PY=f2py3
EXT_SUFFIX := $(shell python3-config --extension-suffix)

default: gaussian_model_simple$(EXT_SUFFIX)

%$(EXT_SUFFIX): %.f90
	$(F2PY) -c -m $* $<

