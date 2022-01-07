from dcwetk.cwe_distance import *
import glob
import pickle
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def getWUMs(tok):
    yrs = []
    wums = []
    master_length = []
    print('getting wums...')
    for name in tqdm(glob.glob('byp_decade_wums/*.pickle')):
        yr = int(name[16:20])
        yrs.append(yr)
        if 1880 <= yr <= 1950:
            with open(name, 'rb') as f:
                w = pickle.load(f)
                f.close()

        if tok in w.keys():
            year_length = 0
            for t, u in w.items():
                year_length += len(u)
                if t == tok:
                    wums.append(u)  # get WUMs

            master_length.append(year_length)   # get total number of tokens in year

    return yrs, wums, master_length


def graph_diachronic_usage(wums, years, tok):
    freqs = list(map(lambda i: len(i.u), wums))

    plt.bar(years, freqs)

    plt.savefig('usage_imgs/' + tok + '.png')

    plt.show()
