from dr import pca
from data.deposit_data_loader import load_cleanse_data
from data.income_evaluation_loader import loadData
from clustering.kmeans import estimate_k, validate_k
from clustering.emclustering import estimate_em_k


def pcakMeans(data, pcVal, label, metric):
    pca_transformed_data = pca.return_principal_components(data, pcVal)
    print('PCA transformed data shape for kmeans ', pca_transformed_data.shape)
    estimate_k({'features': pca_transformed_data}, label, 'plots/dr_clustering/pca/kmeans/', metric)
    validate_k({'features': pca_transformed_data, 'labels': data['labels']}, 'plots/dr_clustering/pca/kmeans/', label)


def pcaem(data, pcVal, label, metric):
    pca_transformed_data = pca.return_principal_components(data, pcVal)
    print('PCA transformed data shape  for em ', pca_transformed_data.shape)
    estimate_em_k({'features': pca_transformed_data}, label, 'plots/dr_clustering/pca/em/', metric)
    validate_k({'features': pca_transformed_data, 'labels': data['labels']}, 'plots/dr_clustering/pca/em/', label)


def run_dr_kmeans():
    # running PCA/kmeans for Deposit Data
    deposit_data = load_cleanse_data()
    pcakMeans(deposit_data, 7, 'deposit', 'manhattan')
    # running PCA/kmeans for income Data
    income_data = loadData()
    pcakMeans(income_data, 4, 'income', 'euclidean')
    # running PCA/em for Deposit Data
    pcaem(deposit_data, 7, 'deposit', 'manhattan')
    # running PCA/em for Deposit Data
    pcaem(income_data, 4, 'income', 'euclidean')
