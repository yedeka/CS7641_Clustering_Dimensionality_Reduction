from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kneed import KneeLocator

def perform_pca(features, datasetLabel):
    data_PCA = PCA(random_state=120)
    data_eigen = data_PCA.fit(features)
    data_variance = data_eigen.explained_variance_
    # variance_total = np.sum(data_variance)
    plot_features = np.arange(start=1, stop=features.shape[1]+1)
    data = {'variance': data_variance,
            'features': plot_features }
    df = pd.DataFrame(data, columns=['variance', 'features'])

    kl = KneeLocator(
        plot_features, data_variance, curve="convex", direction="decreasing"
    )
    print(kl.elbow)
    print(kl.knee)

    kl.plot_knee()

    '''ax = plt.gca()
    df.plot(kind='line', x='features', y='variance', marker='o', ax=ax)'''
    plt.xlabel('Features')
    plt.ylabel('Variance')
    plt.title('Variance vs features')
    plt.grid(True)
    plt.savefig('plots/dr/'+datasetLabel+'_variance_pca.png')
    plt.clf()

def return_principal_components(data, k):
    pca = PCA(random_state=120, n_components=k)
    transformed_data = pca.fit_transform(data['features'])
    return transformed_data
