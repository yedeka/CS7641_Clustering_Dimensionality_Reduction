from data.deposit_data_loader import load_cleanse_data
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier

import numpy as np
import matplotlib.pyplot as plt

def performKmeansNN():
    # load deposit dataset
    data = load_cleanse_data()
    # Select a range of k to check
    target_clusters = [7, 10, 15, 20, 25, 30, 35, 40, 41]
    mlp = MLPClassifier(hidden_layer_sizes=(15, 2), random_state=70, activation='relu', max_iter=500)
    scoring = ['accuracy']
    scores = cross_validate(mlp, data['features'], data['labels'], scoring=scoring, cv=10)
    print(scores)
    NN_fit_time = np.mean(scores['fit_time'])
    NN_accuracy = np.mean(scores['test_accuracy'])


    kmeans_nn_accuracy = []
    kmeans_nn_time = []

    for cluster in target_clusters :
        kmeans = KMeans(n_clusters=cluster, random_state=42)
        clusters = kmeans.fit_predict(data['features'])
        scores = cross_validate(mlp, clusters.reshape(-1, 1), data['labels'], scoring=scoring, cv=10)
        kmeans_nn_accuracy.append(np.mean(scores['test_accuracy']))
        kmeans_nn_time.append(np.mean(scores['fit_time']))

    print(kmeans_nn_accuracy)
    print(kmeans_nn_time)

    plt.style.use("seaborn")
    plt.figure(figsize=(8,8))
    plt.plot(target_clusters, kmeans_nn_accuracy)
    plt.xticks(target_clusters)
    plt.axhline(y=NN_accuracy, color='r', linestyle='-')
    plt.xlabel("# Clusters")
    plt.ylabel('NN Accuracy')
    plt.grid(True)
    plt.savefig('plots/kmeans_nn/deposit/kmeans_nn_accuracy.png')

    plt.clf()

    plt.style.use("seaborn")
    plt.plot(target_clusters, kmeans_nn_time)
    plt.xticks(target_clusters)
    plt.axhline(y=NN_fit_time, color='r', linestyle='-')
    plt.xlabel("Principal Components")
    plt.ylabel('NN Fit Time')
    plt.grid(True)
    plt.savefig('plots/kmeans_nn/deposit/kmeans_nn_fit_time.png')

    plt.clf()

