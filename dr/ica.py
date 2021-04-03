import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import FastICA
from scipy.stats import kurtosis

def perform_ica(features, datasetLabel):
    print('ica start for ',datasetLabel)
    transformer = FastICA( max_iter=550, random_state = 0, whiten=True)
    X_transformed = transformer.fit_transform(features)
    print(X_transformed.shape)
    unmodified_kurtosis = kurtosis(features)
    print('unmodified_kurtosis ',unmodified_kurtosis)
    # An "interface" to matplotlib.axes.Axes.hist() method
    n, bins, patches = plt.hist(x=unmodified_kurtosis, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Kurtosis')
    plt.ylabel('Frequency')
    plt.title(datasetLabel+' Kurtosis Histogram pre ICA')
    plt.show()
    plt.clf()

    X_transformed = pd.DataFrame(X_transformed)
    feature_kurtosis = X_transformed.kurt(axis=0)
    print(feature_kurtosis)
    plt.hist(x=feature_kurtosis, bins='auto', color='#0504aa',
             alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Kurtosis')
    plt.ylabel('Frequency')
    plt.show()
