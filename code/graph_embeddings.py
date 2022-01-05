import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from sklearn.decomposition import PCA


def graphEmbeddings(wums, years):
    wums_pca = []
    labs = []
    for wum, year in zip(wums, years):
        wums_pca.append(PCA(n_components=2).fit_transform(wum))
        labs.append(AffinityPropagation(random_state=10).fit_predict(wum))
