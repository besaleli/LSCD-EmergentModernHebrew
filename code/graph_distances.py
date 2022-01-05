import pandas as pd
import matplotlib.pyplot as plt


def getTokenDF(dfs, years, token):
    filtered_dfs = []
    yrs = []
    for df, year in zip(dfs, years):
        if token in df['token']:
            filtered_dfs.append(df[df['token'] == token])
            yrs.append(year)

    new_df = pd.concat(filtered_dfs, ignore_index=True)
    new_df['year'] = yrs

    return new_df


def graphDistances(dfs, years, token, saveFig=False):
    # instantiate figure
    fig, [prt_ax, jsd_ax, div_ax, apd_ax] = plt.subplots(1, 4)

    # get df of particular token
    tokenDF = getTokenDF(dfs, years, token)

    fig.suptitle('Diachronic LSC of token ' + token[::-1])

    for ax, cat in zip([prt_ax, jsd_ax, div_ax, apd_ax], ['prt', 'jsd', 'div', 'apd']):
        ax.set_title(cat.upper())
        ax.plot(tokenDF['year'], tokenDF[cat], c='black')

    if saveFig:
        plt.savefig('distance_imgs/' + token + '.png')

    plt.show()
