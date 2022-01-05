import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
from sklearn.datasets import make_blobs


def make_IDs(t):
    indices = []
    for j in ([i] * len(t[i]) for i in range(len(t))):
        indices += j

    return indices


"""def get_labs_by_wum(indices, raw_labs):
    N = list(zip(indices, raw_labs))
    labs_by_wum = [[] for _ in range(len(set(indices)))]
    for index, lab in N:
        labs_by_wum[index].append(lab)

    return labs_by_wum"""


"""def get_matrices_by_lab(embeddings, labs):
    labs_set = list(set(labs))
    matrices = []
    for lab in labs_set:
        matrices.append([e for e, l in zip(embeddings, labs) if l == lab])

    return labs_set, matrices"""


"""def prototype(matrix):
    return sum(matrix) / len(matrix)"""


"""def reduce_granularity(embeddings, labs, n_clusters=10):
    ap_labs_set, matrices = get_matrices_by_lab(embeddings, labs)
    matrix_prototypes = [prototype(m) for m in matrices]
    print(len(matrix_prototypes))
    kmeans_labs = KMeans(n_clusters=n_clusters).fit_predict(matrix_prototypes)
    change_rules = {ap_labs_set[i]: kmeans_labs[i] for i in range(len(ap_labs_set))}

    return change_rules"""


class df:
    def __init__(self, u):
        if type(u) == dict:
            self.u_IDs = u['u_IDs']
            self.embeddings = u['embeddings']
            self.kmeans_labs = u['kmeans_labs']
        else:
            self.u_IDs = make_IDs(u)
            self.embeddings = np.concatenate(u)
            self.kmeans_labs = KMeans(n_clusters=5).fit_predict(self.embeddings)

    def partDF(self):
        newDFs = []
        for i in set(self.u_IDs):
            filterCat = lambda cat: [cat[j] for j in range(len(cat)) if self.u_IDs[j] == i]
            newDF = {'u_IDs': filterCat(self.u_IDs),
                     'embeddings': filterCat(self.embeddings),
                     'kmeans_labs': filterCat(self.kmeans_labs)}
            newDFs.append(df(newDF))

        return newDFs

    def freqDist(self):
        return [(i, self.kmeans_labs.count(i) / len(self.kmeans_labs)) for i in set(self.kmeans_labs)]
