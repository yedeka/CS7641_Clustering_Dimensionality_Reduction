from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier


def perform_pca(features, datasetLabel):
    data_PCA = PCA(random_state=120)
    data_eigen = data_PCA.fit(features)
    data_variance = data_eigen.explained_variance_
    plot_features = np.arange(start=1, stop=features.shape[1]+1)
    data = {'variance': data_variance,
            'features': plot_features }
    df = pd.DataFrame(data, columns=['variance', 'features'])

    kl = KneeLocator(
        plot_features, data_variance, curve="convex", direction="decreasing"
    )
    print(kl.elbow)
    kl.plot_knee()
    plt.xlabel('Features')
    plt.ylabel('Variance')
    plt.title('Variance vs features')
    plt.grid(True)
    plt.savefig('plots/dr/pca/'+datasetLabel+'/variance_pca.png')
    plt.clf()

def return_principal_components(data, k):
    pca = PCA(random_state=120, n_components=k)
    transformed_data = pca.fit_transform(data['features'])
    return transformed_data

def validate_pca_nn(data, components, label):

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
        pca = PCA(n_components=component)
        X_transformed = pca.fit_transform(data['features'])
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
    plt.savefig('plots/dr/pca/'+label+'/pca_accuracy.png')

    plt.clf()

    plt.style.use("seaborn")
    plt.plot(components, PCA_fit_time)
    plt.xticks(components)
    plt.axhline(y=NN_fit_time, color='r', linestyle='-')
    plt.xlabel("Principal Components")
    plt.ylabel('NN Fit Time')
    plt.grid(True)
    plt.savefig('plots/dr/pca/' + label + '/pca_fit_time.png')

    plt.clf()

