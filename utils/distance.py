import numpy as np
from scipy.spatial.distance import pdist


def hamming(a, b):
    # compute and return the Hamming distance between the integers
    X=np.vstack([a,b])
    d=pdist(X)[0]
    return d