from functools import wraps


def cached(func):
    cache = {}

    @wraps(func)
    def wrapped(x):
        if x not in cache:
            cache[x] = func(x)
        return cache[x]

    return wrapped
