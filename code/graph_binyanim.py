from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import glob
import pickle
from sklearn.decomposition import PCA
from dcwetk.cwe_distance import *


def getLemmas(fileDir, toks):
    lemmas = []
    for name in tqdm(glob.glob(fileDir)):
        if int(name[-11:-7]) in range(1880, 1951):
            with open(name, 'rb') as f:
                docs = pickle.load(f)
                f.close()

            for doc in docs:
                for sent in doc:
                    for tok in sent:
                        for lem in tok:
                            if lem.lemma in toks:
                                lemmas.append(lem)

    return lemmas


def graphEmbeddings(lemmas, labs, title):
    x, y, z = zip(*PCA(n_components=3).fit_transform(np.array([lem.embedding for lem in lemmas])))
    lemma_labs = [lem.lemma for lem in lemmas]

    df = pd.DataFrame({'token': lemma_labs, 'x': x, 'y': y, 'z': z})
    df['labs'] = list(map(lambda j: labs[j], df['lemma_labs']))

    # instantiate plt figure
    fig = plt.figure()
    ax = fig.add_suplot(projection='3d')
    fig.suptitle(title)

    cmap = plt.cm.viridis

    for i, dff in df.groupby('token'):
        ax.scatter(dff['x'], dff['y'], dff['z'], c=cmap(dff['token']),
                   label=dff['token'])

    plt.legend()

