import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from math import sqrt

from sklearn.decomposition import FastICA
from scipy.stats import kurtosis
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier

def perform_ica(features, datasetLabel, components):
    print('ica start for ',datasetLabel)
    transformer = FastICA( max_iter=550, random_state = 0, whiten=True)
    X_transformed = transformer.fit_transform(features)
    print(X_transformed.shape)
    unmodified_kurtosis = kurtosis(features)
    print('Unmodified_kurtosis ',unmodified_kurtosis)
    X_transformed = pd.DataFrame(X_transformed)
    feature_kurtosis = X_transformed.kurt(axis=0)
    print('Modified_kurtosis ',feature_kurtosis)
    recon_error = []
    # validate RMSE for reconstruction
    for component in components:
        transformer = FastICA(max_iter=550, random_state=20, whiten=True, n_components=component)
        X_transformed = transformer.fit_transform(features)
        X_recon = transformer.inverse_transform(X_transformed)
        rmse = sqrt(mean_squared_error(features, X_recon))
        recon_error.append(rmse)

    print('recon_error => ',recon_error)

    plt.style.use("seaborn")
    plt.plot(components, recon_error, marker='o')
    plt.xticks(components, rotation="90")
    plt.xlabel("ICA Components")
    plt.ylabel('Reconstruction error')
    plt.savefig('plots/dr/ica/' + datasetLabel + '/ica_recon_error.png')
    plt.clf()

def validate_ica_nn(data, components, label):

    mlp = MLPClassifier(hidden_layer_sizes=(15, 2), random_state=70, activation='relu', max_iter=500)
    scoring = ['accuracy']
    scores = cross_validate(mlp, data['features'], data['labels'], scoring=scoring, cv=10)
    print(scores)
    NN_fit_time = np.mean(scores['fit_time'])
    NN_accuracy = np.mean(scores['test_accuracy'])
    print(NN_fit_time)
    print(NN_accuracy)

    ICA_fit_time = []
    ICA_accuracy = []
    for component in components:
        pca = FastICA(n_components=component)
        X_transformed = pca.fit_transform(data['features'])
        scores_pca = cross_validate(mlp, X_transformed, data['labels'], scoring=scoring, cv=10)
        print(scores_pca)
        ICA_fit_time.append(np.mean(scores_pca['fit_time']))
        ICA_accuracy.append(np.mean(scores_pca['test_accuracy']))


    print('ICA_fit_time => ',ICA_fit_time)
    print('ICA_accuracy => ', ICA_accuracy)

    plt.style.use("seaborn")
    plt.figure(figsize=(8,8))
    plt.plot(components, ICA_accuracy, marker='o')
    plt.xticks(components)
    plt.axhline(y=NN_accuracy, color='r', linestyle='-')
    plt.xlabel("Principal Components")
    plt.ylabel('NN Accuracy')
    plt.grid(True)
    plt.savefig('plots/dr/ica/'+label+'/ica_accuracy.png')

    plt.clf()

    plt.style.use("seaborn")
    plt.plot(components, ICA_fit_time, marker='o')
    plt.xticks(components)
    plt.axhline(y=NN_fit_time, color='r', linestyle='-')
    plt.xlabel("Principal Components")
    plt.ylabel('NN Fit Time')
    plt.grid(True)
    plt.savefig('plots/dr/ica/' + label + '/ica_fit_time.png')

    plt.clf()