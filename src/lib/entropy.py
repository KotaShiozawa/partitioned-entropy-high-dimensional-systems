import numpy as np
import pandas as pd
from scipy.stats import entropy

def entropy_time_evolution(labels, tau_span):
    _, freq = np.unique(labels, return_counts=True)
    tau_end = tau_span[-1] + 1
    labels0 = labels[:-tau_end]
    transition = labels0[1:]!=labels0[:-1]
    S0 = labels0[1:][transition]

    h = []
    for tau in tau_span:
        S = labels[tau+1:-tau_end+tau][transition]
        tab = pd.crosstab(S0, S)
        e = entropy(tab, axis=1)
        h.append(sum(e*freq/sum(freq)))
    return np.array(h)


def symbol(X):
    m = np.median(X, axis=0)
    S = [''.join(np.array((x > m) + 0, dtype='str')) for x in X]
    return np.array(S)
