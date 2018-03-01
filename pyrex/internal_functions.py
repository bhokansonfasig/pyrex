"""Helper functions for use in PyREx modules."""

import copy
import numpy as np

def normalize(vector):
    """Returns the normalized form of the given vector."""
    v = np.array(vector)
    mag = np.linalg.norm(v)
    if mag==0:
        return v
    else:
        return v / mag



# Note: using lazy_property decorator instead of simple python property
# decorator incerases access time for property (after initial calculation)
# from ~0.5 microseconds to ~5 microseconds, so it is only recommended on
# properties with calculation times >5 microseconds which will be accessed
# more than once
def lazy_property(fn):
    """Decorator that makes a property lazily evaluated."""
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property


# Note: additional time added in setting attribute of lazy_mutable_class
# has not been tested, but it shouldn't be significant compared to the time
# saved in lazy_evaluation of properties
def lazy_mutable_class(*static_attributes):
    """Decorator which allows some class properties to be lazily evaluated as
    long as the specified static attributes are not changed."""
    def lazy_class_decorator(cls):
        """Decorator for lazy class"""
        class LazyClass:
            def __init__(self, *args, **kwargs):
                self._wrapped = cls(*args, **kwargs)

            def __getattribute__(self, name):
                if name=="_wrapped":
                    return super().__getattribute__(name)
                else:
                    return self._wrapped.__getattribute__(name)

            def __setattr__(self, name, value):
                if name in static_attributes:
                    # print("Clearing all lazy attributes")
                    lazy_attributes = [attr for attr in self._wrapped.__dict__
                                       if attr.startswith("_lazy_")]
                    for lazy_attr in lazy_attributes:
                        delattr(self._wrapped, lazy_attr)

                super().__setattr__(name, value)

        return LazyClass

    return lazy_class_decorator
