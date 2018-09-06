class NoLogger(object):

    """Logger object with methods debug, info, error, fatal and warn
    which ignore logging messages.

    Used as default loggerin inferences.py to supress logging messages.
    """

    def _ignore(self, *a, **kw):
        pass

    debug = info = error = fatal = warn = _ignore
