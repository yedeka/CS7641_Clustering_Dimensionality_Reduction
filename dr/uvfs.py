from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
import numpy as np


def apply_uvfs(data, indicator,componentVal):
    X_new = SelectKBest(f_classif, k=componentVal).fit_transform(data['features'], data['labels'])
    print(indicator)
    print("X_new dimensions => ", X_new.shape)
    return X_new

def validate_uvfs_nn(data,components,label):
    mlp = MLPClassifier(hidden_layer_sizes=(15, 2), random_state=70, activation='relu', max_iter=500)
    scoring = ['accuracy']
    scores = cross_validate(mlp, data['features'], data['labels'], scoring=scoring, cv=10)
    print(scores)
    NN_fit_time = np.mean(scores['fit_time'])
    NN_accuracy = np.mean(scores['test_accuracy'])
    print(NN_fit_time)
    print(NN_accuracy)

    PCA_fit_time = []
    PCA_accuracy = []
    for component in components:
        X_transformed = SelectKBest(f_classif, k=component).fit_transform(data['features'], data['labels'])
        scores_pca = cross_validate(mlp, X_transformed, data['labels'], scoring=scoring, cv=10)
        print(scores_pca)
        PCA_fit_time.append(np.mean(scores_pca['fit_time']))
        PCA_accuracy.append(np.mean(scores_pca['test_accuracy']))

    plt.style.use("seaborn")
    plt.figure(figsize=(8, 8))
    plt.plot(components, PCA_accuracy)
    plt.xticks(components)
    plt.axhline(y=NN_accuracy, color='r', linestyle='-')
    plt.xlabel("UVFS Components")
    plt.ylabel('NN Accuracy')
    plt.grid(True)
    plt.savefig('plots/dr/uvfs/' + label + '/uvfs_accuracy.png')

    plt.clf()

    plt.style.use("seaborn")
    plt.plot(components, PCA_fit_time)
    plt.xticks(components)
    plt.axhline(y=NN_fit_time, color='r', linestyle='-')
    plt.xlabel("UVFS Components")
    plt.ylabel('NN Fit Time')
    plt.grid(True)
    plt.savefig('plots/dr/uvfs/' + label + '/uvfs_fit_time.png')

    plt.clf()