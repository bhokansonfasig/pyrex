"""
Helper functions and classes for use in PyREx modules.

This module is intended as a container for functions, typically used in more
than one PyREx module, which are not physics-motivated and are instead used
mainly to clean up code. Functions and classes in this module may also be
computer-science-motivated structures that python doesn't include naturally.

"""

from collections.abc import Iterable
import copy
import functools
import logging
import numpy as np

logger = logging.getLogger(__name__)


def normalize(vector):
    """
    Normalize the given vector.

    Parameters
    ----------
    vector : array_like

    Returns
    -------
    ndarray
        Normalized form of `vector`.

    Examples
    --------
    >>> normalize([5,0,0])
    array([1., 0., 0.])

    >>> v = np.array([1,0,1])
    >>> normalize(v)
    array([0.70710678, 0.        , 0.70710678])

    """
    v = np.array(vector)
    mag = np.linalg.norm(v)
    if mag==0:
        return v
    else:
        return v / mag


def get_from_enum(value, enum):
    """
    Find the enum value given some representation of it.

    Transforms the given `value` into the corresponding value from the `enum`
    by checking the type of `value` given.

    Parameters
    ----------
    value
        Representation of the desired `enum` value. If already a member of
        `enum`, no change. If ``str``, assumed to be a name in the `enum`.
        Otherwise, assumed to be a value type of the `enum`.
    enum : Enum
        Python ``Enum`` to compare names values with.

    Returns
    -------
    Enum value
        Value in the `enum` represented by the given `value`.

    Examples
    --------
    >>> from enum import Enum
    >>> class Color(Enum):
    ...     red = 1
    ...     green = 2
    ...     blue = 3
    >>> get_from_enum(Color.red, Color)
    <Color.red: 1>
    >>> get_from_enum("green", Color)
    <Color.green: 2>
    >>> get_from_enum(3, Color)
    <Color.blue: 3>

    """
    if isinstance(value, enum):
        return value
    elif isinstance(value, str):
        return enum[value]
    else:
        return enum(value)


def flatten(iterator, dont_flatten=()):
    """
    Flattens an iterator to iterate over all elements individually.

    Flattens all iterable elements in the given iterator recursively and
    yields the resulting flat iterator. Can optionally not flatten certain
    classes. Will not flatten strings or bytes to avoid recursion errors.

    Parameters
    ----------
    iterator : iterable object
        Iterable object to flatten.
    dont_flatten : tuple_like, optional
        Tuple (or similar) of classes which should not be flattened.

    Yields
    ------
    element : any
        Each element of `iterator` with sub-iterators expanded out.

    Notes
    -----
    Since ``str`` and ``bytes`` objects are always considered iterable despite
    their length, these objects will not be flattened and will remain intact.

    If a class is asked not to be flattened, any sub-iterators contained in an
    iterator of that class will not be flattened either (see examples).

    Examples
    --------
    >>> list(flatten([1, 2, (3, 'four', [5, 6], 7), [8, 9]]))
    [1, 2, 3, 'four', 5, 6, 7, 8, 9]

    >>> list(flatten([1, 2, (3, 'four', [5, 6], 7), [8, 9]], dont_flatten=(tuple,)))
    [1, 2, (3, 'four', [5, 6], 7), 8, 9]

    >>> list(flatten([1, 2, [3, 'four', (5, 6), 7], [8, 9]], dont_flatten=(tuple,)))
    [1, 2, 3, 'four', (5, 6), 7, 8, 9]

    """
    for element in iterator:
        if (isinstance(element, Iterable) and
                not isinstance(element, tuple(dont_flatten)+(str, bytes))):
            yield from flatten(element, dont_flatten=dont_flatten)
        else:
            yield element



def complex_interp(x, xp, fp, method='cartesian', outer=None):
    """
    Perform interpolation on complex values.

    Calculates the linear interpolation of an array of complex values `fp`.
    Interpolates in real and imaginary parts for the Cartesian method, and
    interpolates in gain and (unwrapped) phase for the Euler method.

    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.
    xp : array_like
        The x-coordinates of the data points, must be increasing.
    fp : array_like
        The (complex-valued) y-coordinates of the data points, same length as
        `xp`.
    method : {'cartesian', 'euler'}, optional
        The interpolation method to use between data points. 'Cartesian' will
        interpolate linearly in the real/imaginary plane. 'Euler' will
        interpolate linearly in the gain and (unwrapped) phase values.
    outer : None or float, optional
        The value to use for values of `x` outside of the range of `xp`. In the
        'Cartesian' method it will be applied as a complex value. In the
        'Euler' method it will be applied to the gain, but phases will always
        retain their value at the closest edge of `xp`.

    Returns
    -------
    y : ndarray
        The (complex-valued) interpolated values, same shape as `x`.

    Raises
    ------
    ValueError
        If an unspecified interpolation method is given.
        Or if `xp` and `fp` have different length.
        Or if `xp` and `fp` are not 1-D sequences.

    """
    if method.lower()=='cartesian':
        return np.interp(x, xp, fp, left=outer, right=outer)
    elif method.lower()=='euler':
        gain = np.interp(x, xp, np.abs(fp), left=outer, right=outer)
        phase = np.interp(x, xp, np.unwrap(np.angle(fp)))
        return gain * np.exp(1j*phase)
    else:
        raise ValueError("Unknown interpolation method '"+str(method)+"'")


def complex_bilinear_interp(x, y, xp, yp, fp, method='cartesian',
                            unwrap_axis=0):
    """
    Perform bilinear interpolation on complex values.

    Calculates the bilinear interpolation of an array of complex values `fp`.
    Interpolates in real and imaginary parts for the Cartesian method, and
    interpolates in gain and (unwrapped) phase for the Euler method.

    Parameters
    ----------
    x : float
        The x-coordinate at which to evaluate the interpolated values.
    y : float
        The y-coordinate at which to evaluate the interpolated values.
    xp : array_like
        The x-coordinates of the data points, must be increasing.
    yp : array_like
        The y-coordinates of the data points, must be increasing.
    fp : array_like
        The (complex-valued) z-coordinates of the data points, with its second
        dimension the same length as `xp` and its third dimension the same
        length as `yp`.
    method : {'cartesian', 'euler'}, optional
        The interpolation method to use between data points. 'Cartesian' will
        interpolate linearly in the real/imaginary plane. 'Euler' will
        interpolate linearly in the gain and (unwrapped) phase values.
    unwrap_axis : int, optional
        When used with the 'euler' method, controls which axis of `fp` the
        phases are unwrapped around.

    Returns
    -------
    z : ndarray
        The (complex-valued) interpolated values.

    Raises
    ------
    ValueError
        If an unspecified interpolation method is given.
        Or if `x` or `y` lies outside of the range of `xp` or `yp`.
        Or if the shape of `fp` is incompatible with `xp` and `yp`.

    """
    if len(xp)!=fp.shape[1] or len(yp)!=fp.shape[2]:
        raise ValueError("Dimensions of fp "+str(fp.shape)+" are incompatible "
                         +"with xp ("+len(xp)+") and/or yp ("+len(yp)+")")
    if not xp[0]<=x<=xp[-1]:
        raise ValueError("x value "+str(x)+" outside of the range of xp")
    i = np.searchsorted(xp, x)
    if i==0:
        i = 1
    i_low = i-1
    i_high = i
    t = (x - xp[i_low]) / (xp[i_high] - xp[i_low])
    if not yp[0]<=y<=yp[-1]:
        raise ValueError("y value "+str(y)+" outside of the range of yp")
    j = np.searchsorted(yp, y)
    if j==0:
        j = 1
    j_low = j - 1
    j_high = j
    u = (y - yp[j_low]) / (yp[j_high] - yp[j_low])
    def interp(f):
        return (
            (1-t)*(1-u) * f[:, i_low,  j_low ] +
               t *(1-u) * f[:, i_high, j_low ] +
            (1-t)*   u  * f[:, i_low,  j_high] +
               t *   u  * f[:, i_high, j_high]
        )
    if method.lower()=='cartesian':
        return interp(fp)
    elif method.lower()=='euler':
        gain = interp(np.abs(fp))
        phase = interp(np.unwrap(np.angle(fp), axis=unwrap_axis))
        return gain * np.exp(1j*phase)



def mirror_func(match_func, run_func, self=None):
    """
    Mirror the attributes of one function onto another.

    Creates a function which operates like one function, but has all the
    attributes of another. Works for functions or class methods.

    Parameters
    ----------
    match_func : function
        Function with the attributes to be mirrored.
    run_func : function
        Function with the desired behavior.
    self : object or None, optional
        If ``None``, `run_func` called as a regular function, otherwise
        `run_func` is called as a class method (with `self` as its first
        argument).

    Returns
    -------
    function
        Function with the behavior of `run_func`, but the attributes of
        `match_func`.

    Examples
    --------
    >>> from inspect import signature
    >>> def descriptive_add(a, b):
    ...     \"\"\"Function with a descriptive docstring.\"\"\"
    ...     pass
    >>> def add_implementation(x, y):
    ...     # Actually adds, but no docs or anything
    ...     return x+y
    >>> my_add = mirror_func(descriptive_add, add_implementation)
    >>> my_add(2, 3)
    5
    >>> my_add.__doc__
    'Function with a descriptive docstring.'
    >>> signature(my_add)
    <Signature (a, b)>

    >>> from inspect import signature
    >>> class A:
    ...     def __init__(self, value):
    ...         self.value = value
    ...     def mult(self, factor, power=1):
    ...         \"\"\"Multiplies value by factor and raises to power.\"\"\"
    ...         return (self.value*factor)**power
    >>> class B(A):
    ...     def __init__(self, value):
    ...         self.value = value
    ...         # Make the mult method look the same as for A, but with
    ...         # different behavior
    ...         self.mult = mirror_func(A.mult, B.different_mult, self=self)
    ...     def different_mult(self, *args, **kwargs):
    ...         \"\"\"Different implementation of mult.\"\"\"
    ...         return (self.value*int(args[0]))**kwargs['power']
    >>> b = B(5)
    >>> b.mult(2.5, power=2)
    100
    >>> b.mult.__doc__
    'Multiplies by factor and raises to power.'
    >>> signature(b.mult)
    <Signature (self, factor, power=1)>

    """
    logger.debug("Mirroring %s onto %s", match_func, run_func)
    @functools.wraps(match_func)
    def wrapper(*args, **kwargs):
        if self is not None:
            return run_func(self, *args, **kwargs)
        else:
            return run_func(*args, **kwargs)
    return wrapper


def lazy_property(fn):
    """
    Decorator that makes a property lazily evaluated.

    Acts like the standard python ``property`` decorator, but the first time
    the decorated property is accessed an attribute with the property's name
    prefixed by '_lazy_' will be created and the value of the property will be
    stored. Upon further access of the property, the stored value will be
    returned instead of recalculating it.

    Parameters
    ----------
    fn : function
        Function returning class property which is to be decorated.

    Returns
    -------
    function
        Lazy-evaluation property function.

    See Also
    --------
    LazyMutableClass : Class for lazy properties dependent on attributes.

    Notes
    -----
    Using the ``lazy_property`` decorator instead of the simple python
    ``property`` decorator increases the time for property access (after the
    initial calculation) from ~0.5 microseconds to ~5 microseconds, so
    ``lazy_property`` is only recommended for use on properties with
    calculation times >5 microseconds which are likely to be accessed more than
    once.

    Examples
    --------
    >>> from time import sleep
    >>> class A:
    ...     def __init__(self, value):
    ...         self.value = value
    ...     @lazy_property
    ...     def twice(self):
    ...         sleep(5)
    ...         return self.value*2
    >>> a = A(1)
    >>> "_lazy_twice" in a.__dict__
    False
    >>> a.twice
    2
    >>> "_lazy_twice" in a.__dict__
    True
    >>> a.twice
    2

    """
    attr_name = '_lazy_' + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property


# Note: additional time added in setting attribute of LazyMutableClass
# has not been tested, but it shouldn't be significant compared to the time
# saved in lazy evaluation of properties
class LazyMutableClass:
    """
    Class with lazy properties which may depend on other class attributes.

    This class is intended as a base class for any class which desires lazy
    properties which depend on other attributes and thus may need to be
    recalculated when the class attributes change. Any lazy properties in this
    class will be lazily evaluated as usual until one of the given static
    attributes changes, at which point all lazy properties will be cleared and
    will be recalculated on their next call. By default the static attributes
    of the class will be set to all attributes present at the time of the
    ``LazyMutableClass.__init__`` call.

    Parameters
    ----------
    static_attributes : None or sequence of str, optional
        Set of attribute names on which the lazy properties depend. If ``None``
        then it will contain all members of ``__dict__`` at the time of the
        call.

    See Also
    --------
    lazy_property : Decorator for lazily-evaluated properties.

    Examples
    --------
    >>> from time import sleep
    >>> class A(LazyMutableClass):
    ...     def __init__(self, value):
    ...         self.value = value
    ...         super().__init__()
    ...     @lazy_property
    ...     def twice(self):
    ...         sleep(5)
    ...         return self.value*2
    >>> a = A(1)
    >>> "_lazy_twice" in a.__dict__
    False
    >>> a.twice
    2
    >>> "_lazy_twice" in a.__dict__
    True
    >>> a.twice
    2
    >>> a.value = 5
    >>> "_lazy_twice" in a.__dict__
    False
    >>> a.twice
    10
    >>> "_lazy_twice" in a.__dict__
    True
    >>> a.twice
    10

    """
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
        # If static attributes have not yet been set, just use default setattr.
        # This avoids problems with setting _static_attrs in the first place.
        if "_static_attrs" in self.__dict__ and name in self._static_attrs:
            self._clear_cache()
        super().__setattr__(name, value)

    def _clear_cache(self):
        """Clears the cache of lazily-evaluated parameters."""
        lazy_attributes = [attr for attr in self.__dict__
                           if attr.startswith("_lazy_")]
        for lazy_attr in lazy_attributes:
            delattr(self, lazy_attr)
