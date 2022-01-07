import matplotlib.pyplot as plt


def graph_diachronic_usage(wums, years, tok):
    freqs = list(map(lambda i: len(i.u), wums))

    plt.bar(years, freqs)

    plt.savefig('usage_imgs/' + tok + '.png')

    plt.show()
