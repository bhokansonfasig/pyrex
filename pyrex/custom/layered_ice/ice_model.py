"""
Module containing ice model classes.

Includes ice models for layered ice.

"""

import logging
import collections
import numpy as np

logger = logging.getLogger(__name__)


class LayeredIce:
    """
    Class describing ice divided into multiple layers.

    Supports building layers made of any typical ice model. Most methods
    dispatch to the matching methods of the layers and combine the results
    appropriately. In all methods, the depth z should be given as a negative
    value if it is below the surface of the ice.

    Parameters
    ----------
    layers : array_like
        Array containing the ice models for each layer of the ice. Each ice
        model should have an `index` method for calculating the index of
        refraction at a depth as well as a `range` attribute describing the
        depth range of the ice layer.
    index_above : float or None, optional
        Index of refraction above the uppermost ice layer. If `None`, uses the
        index of refraction at the uppermost boundary.
    index_below : float or None, optional
        Index of refraction below the lowermost ice layer. If `None`, uses the
        index of refraction at the lowermost boundary.

    Attributes
    ----------
    layers : list
        List of ice models for each layer of the ice.
    index_above : float or None, optional
        Index of refraction above the uppermost ice layer.
    index_below : float or None, optional
        Index of refraction below the lowermost ice layer.

    """
    def __init__(self, layers, index_above=1, index_below=None):
        self.layers = list(sorted(layers, key=lambda x: -x.valid_range[0]))
        self._index_above = index_above
        self._index_below = index_below

    @property
    def index_above(self):
        """The index of refraction above the ice's valid range."""
        if self._index_above is None:
            return self.layers[0].index(self.layers[0].valid_range[1])
        else:
            return self._index_above

    @index_above.setter
    def index_above(self, index):
        self._index_above = index

    @property
    def index_below(self):
        """The index of refraction below the ice's valid range."""
        if self._index_below is None:
            return self.layers[-1].index(self.layers[-1].valid_range[0])
        else:
            return self._index_below

    @index_below.setter
    def index_below(self, index):
        self._index_below = index

    def contains(self, point):
        """
        Determines if the given point is within the ice's valid range.

        Parameters
        ----------
        point : array_like
            Point to be tested.

        Returns
        -------
        bool
            Whether `point` is cointained within the ice model.

        """
        for layer in self.layers:
            if layer.contains(point):
                return True
        return False

    @property
    def boundaries(self):
        """
        List of the depths which separate the layers of the ice.

        Includes the upper bound of the uppermost layer and the lower bound of
        the lowermost layer.

        """
        strata = [self.layers[0].valid_range[1], self.layers[0].valid_range[0]]
        for layer in self.layers[1:]:
            if layer.valid_range[1]!=strata[-1]:
                raise ValueError("Adjacent layers do not connect properly")
            strata.append(layer.valid_range[0])
        return strata

    def layer_at_depth(self, z):
        """
        Determines the layer of ice at a given depth.

        Supports passing an array of depths.

        Parameters
        ----------
        z : array_like
            (Negative-valued) depths (m) in the ice.

        Returns
        -------
        array_like
            Ice models of the layers at `z` values.

        """
        if not isinstance(z, collections.Iterable):
            single_value = True
            z = [z]
        else:
            single_value = False

        layers = []
        for depth in z:
            for layer in self.layers:
                if layer.valid_range[0]<depth<=layer.valid_range[1]:
                    layers.append(layer)
                    break
            else:
                if layer.valid_range[0]==depth:
                    layers.append(layer)
                else:
                    raise ValueError("No layer at depth "+str(depth))

        if single_value:
            return layers[0]
        else:
            return layers

    def index(self, z):
        """
        Calculates the index of refraction of the ice at a given depth.

        Dispatches the index of refraction calculations to the appropriate
        layers. Supports passing an array of depths.

        Parameters
        ----------
        z : array_like
            (Negative-valued) depths (m) in the ice.

        Returns
        -------
        array_like
            Indices of refraction at `z` values.

        """
        if not isinstance(z, collections.Iterable):
            single_value = True
            z = [z]
        else:
            single_value = False

        indices = []
        for depth in z:
            try:
                n = self.layer_at_depth(depth).index(depth)
            except ValueError:
                if depth>self.layers[0].valid_range[1]:
                    n = self.index_above
                elif depth<=self.layers[-1].valid_range[0]:
                    n = self.index_below
                else:
                    raise ValueError("No index at depth "+str(depth))
            indices.append(n)

        if single_value:
            return indices[0]
        else:
            return np.asarray(indices)

    def depth_with_index(self, n):
        """
        Calculates the corresponding depth for a given index of refraction.

        Not implemented for layered ice.

        Parameters
        ----------
        n : array_like
            Indices of refraction.

        Returns
        -------
        array_like
            (Negative-valued) depths corresponding to the given `n` values.

        """
        raise NotImplementedError("depth_with_index is unavailable for "+
                                  self.__class__.__name__)
