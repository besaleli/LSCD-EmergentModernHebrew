import numpy as np
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine
import torch
import math
from random import sample

prototype = lambda i: sum(i) / len(i)
xorCond = lambda i, j: i is None and j is not None
paramCond = lambda i, j: xorCond(i, j) or xorCond(j, i) or (i is None and j is None)


def sample_Uw(w, sample_size=0.5, min_sample_size=10):
    n_vecs = math.ceil(len(w) * sample_size)
    return np.array(sample(w, n_vecs)) if n_vecs >= min_sample_size else w


def standardize(w):
    p = prototype(w)
    w_no_mean = np.array([i - p for i in w])
    w_scaled = normalize(w_no_mean, norm='l1', axis=0)

    return w_scaled


def prt(u1, u2):
    u1s, u2s = standardize(u1), standardize(u2)
    p1, p2 = prototype(u1s), prototype(u2s)

    return 1 / cosine(p1, p2)


def apd(u1, u2, sample_size=None, min_sample_size=10, device=None, max_sample_size=1024):
    u1s, u2s = standardize(u1), standardize(u2)
    torchCos = torch.nn.CosineSimilarity(dim=1)
    dist_from_sim = lambda i: 1 - i
    toTorch = lambda i: torch.tensor(np.array(i), device=device) if device else torch.tensor(np.array(i))

    # only sample or max_sample_size or none can have a value
    assert paramCond(sample_size, max_sample_size)

    # sample if necessary
    if sample_size is not None:
        samp = lambda i: sample_Uw(i, sample_size=sample_size, min_sample_size=min_sample_size)
    elif max_sample_size is not None:
        samp = lambda i: sample(i, max_sample_size) if len(i) > max_sample_size else i
    else:
        samp = lambda i: i

    prevSample, currSample = samp(list(u1s)), samp(list(u2s.u))

    arr1, arr2 = [], []
    for x, y in ((x, y) for x in prevSample for y in currSample):
        arr1.append(x)
        arr2.append(y)

    arr1_t, arr2_t = toTorch(arr1), toTorch(arr2)

    distances = list(map(dist_from_sim, torchCos(arr1_t, arr2_t)))

    apd_dist = sum(distances) / len(distances)

    del arr1_t, arr2_t

    return float(apd_dist)


def div(u1, u2):
    u1s, u2s = standardize(u1), standardize(u2)
    p1, p2 = prototype(u1s), prototype(u2s)

    d = lambda v: cosine(v, p1)
    dists_from_p1 = list(map(d, u1s))
    dists_from_p2 = list(map(d, u2s))

    var_coefficient_1 = sum(dists_from_p1) / len(u1)
    var_coefficient_2 = sum(dists_from_p2) / len(u2)

    return abs(var_coefficient_1 - var_coefficient_2)
