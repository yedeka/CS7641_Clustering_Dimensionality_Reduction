from sklearn import random_projection
from clustering.kmeans import validate_k_fixed
from clustering.emclustering import validate_em_k_fixed
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier

import matplotlib.pyplot as plt
import numpy as np


def return_random_components(data, components):
    transformer = random_projection.SparseRandomProjection(n_components=components, random_state=150)
    X_transformed = transformer.fit_transform(data['features'])
    return X_transformed

def apply_rp(data, components,indicator, k1, k2):
    print('Random projection for ',indicator)
    rand_kmean_scores = []
    rand_emm_scores = []

    # validate RMSE for reconstruction
    for component in components:
        transformer = random_projection.SparseRandomProjection(n_components=component, random_state=150)
        X_transformed = transformer.fit_transform(data['features'])
        rand_kmean_scores.append(validate_k_fixed(X_transformed, data['labels'], k1))
        rand_emm_scores.append(validate_em_k_fixed(X_transformed, data['labels'], k2))

    print('k means adj rand scores => ',rand_kmean_scores)

    plt.style.use("seaborn")
    plt.plot(components, rand_kmean_scores, marker='o')
    plt.xticks(components, rotation="90")
    plt.xlabel("RP Components")
    plt.ylabel('Adjusted rand scores')
    plt.savefig('plots/dr/rp/' + indicator + '/kmeans/rp_adj_rand_scores.png')
    plt.clf()

    print('em adj rand scores => ', rand_emm_scores)

    plt.style.use("seaborn")
    plt.plot(components, rand_emm_scores, marker='o')
    plt.xticks(components, rotation="90")
    plt.xlabel("RP Components")
    plt.ylabel('Adjusted rand scores')
    plt.savefig('plots/dr/rp/' + indicator + '/em/rp_adj_rand_scores.png')
    plt.clf()

def validate_rp_nn(data, components, label):

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
        rp = random_projection.SparseRandomProjection(n_components=component, random_state=150)
        X_transformed = rp.fit_transform(data['features'])
        scores_pca = cross_validate(mlp, X_transformed, data['labels'], scoring=scoring, cv=10)
        print(scores_pca)
        PCA_fit_time.append(np.mean(scores_pca['fit_time']))
        PCA_accuracy.append(np.mean(scores_pca['test_accuracy']))

    plt.style.use("seaborn")
    plt.figure(figsize=(8,8))
    plt.plot(components, PCA_accuracy)
    plt.xticks(components)
    plt.axhline(y=NN_accuracy, color='r', linestyle='-')
    plt.xlabel("Principal Components")
    plt.ylabel('NN Accuracy')
    plt.grid(True)
    plt.savefig('plots/dr/rp/'+label+'/rp_accuracy.png')

    plt.clf()

    plt.style.use("seaborn")
    plt.plot(components, PCA_fit_time)
    plt.xticks(components)
    plt.axhline(y=NN_fit_time, color='r', linestyle='-')
    plt.xlabel("Principal Components")
    plt.ylabel('NN Fit Time')
    plt.grid(True)
    plt.savefig('plots/dr/rp/' + label + '/rp_fit_time.png')

    plt.clf()
