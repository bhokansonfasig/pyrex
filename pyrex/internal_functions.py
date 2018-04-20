"""Helper functions for use in PyREx modules."""

import collections
import copy
import logging
import numpy as np

logger = logging.getLogger(__name__)


def normalize(vector):
    """Returns the normalized form of the given vector."""
    v = np.array(vector)
    mag = np.linalg.norm(v)
    if mag==0:
        return v
    else:
        return v / mag


def flatten(iterator, dont_flatten=()):
    """Flattens all iterable elements in the given iterator recursively and
    returns the resulting flat iterator. Can optionally be passed a list of
    classes to avoid flattening. Will not flatten strings or bytes due to
    recursion errors."""
    for element in iterator:
        if (isinstance(element, collections.Iterable) and
                not isinstance(element, dont_flatten+(str, bytes))):
            yield from flatten(element, dont_flatten=dont_flatten)
        else:
            yield element


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
            # print("Setting", attr_name)
        return getattr(self, attr_name)

    return _lazy_property


# Note: additional time added in setting attribute of LazyMutableClass
# has not been tested, but it shouldn't be significant compared to the time
# saved in lazy evaluation of properties
class LazyMutableClass:
    """Class whose properties can be lazily evaluated by using lazy_property
    decorator, but will re-evaluate lazy properties if any of its specified
    static_attributes change. By default, static_attributes is set to all
    attributes of the class at the time of the init call."""
    def __init__(self, static_attributes=None):
        # If static_attributes not specified, set to any currently-set attrs
        # Allows for easy setting of static attributes in subclasses
        # by simply delaying the super().__init__ call
        if static_attributes is None:
            self._static_attrs = [attr for attr in self.__dict__
                                  if not attr.startswith("_")]
        else:
            self._static_attrs = static_attributes

    def __setattr__(self, name, value):
        # If static attributes have not yet been set, just use default setattr
        if "_static_attrs" not in self.__dict__:
            super().__setattr__(name, value)

        elif name in self._static_attrs:
            lazy_attributes = [attr for attr in self.__dict__
                               if attr.startswith("_lazy_")]
            for lazy_attr in lazy_attributes:
                # print("Clearing", lazy_attr)
                delattr(self, lazy_attr)

        super().__setattr__(name, value)
