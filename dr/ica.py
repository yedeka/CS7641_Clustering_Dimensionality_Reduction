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
    print('Unmodified_kurtosis ',unmodified_kurtosis)
    X_transformed = pd.DataFrame(X_transformed)
    feature_kurtosis = X_transformed.kurt(axis=0)
    print('Modified_kurtosis ',feature_kurtosis)
