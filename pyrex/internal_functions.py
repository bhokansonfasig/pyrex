"""Helper functions for use in PyREx modules."""

import numpy as np

def normalize(vector):
    v = np.array(vector)
    mag = np.linalg.norm(v)
    if mag==0:
        return v
    else:
        return v / mag
