from dr import pca, ica, randomprojection, uvfs
from data.deposit_data_loader import load_cleanse_data
from data.income_evaluation_loader import loadData
from clustering.kmeans import estimate_k, validate_k
from clustering.emclustering import estimate_em_k, validate_em_k


def pcakMeans(data, pcVal, label, metric):
    pca_transformed_data = pca.return_principal_components(data, pcVal)
    print('PCA transformed data shape for kmeans ', pca_transformed_data.shape)
    estimate_k({'features': pca_transformed_data}, label, 'plots/dr_clustering/pca/kmeans/', metric)
    validate_k({'features': pca_transformed_data, 'labels': data['labels']}, 'plots/dr_clustering/pca/kmeans/', label)


def pcaem(data, pcVal, label, metric):
    pca_transformed_data = pca.return_principal_components(data, pcVal)
    print('PCA transformed data shape  for em ', pca_transformed_data.shape)
    estimate_em_k({'features': pca_transformed_data}, label, 'plots/dr_clustering/pca/em/', metric)
    validate_em_k({'features': pca_transformed_data, 'labels': data['labels']}, 'plots/dr_clustering/pca/em/', label)

def icakMeans(data, icVal,label, metric):
    ica_transformed_data = ica.return_independent_components(data, icVal)
    print('ICA transformed data shape for kmeans ', ica_transformed_data.shape)
    estimate_k({'features': ica_transformed_data}, label, 'plots/dr_clustering/ica/kmeans/', metric)
    validate_k({'features': ica_transformed_data, 'labels': data['labels']}, 'plots/dr_clustering/ica/kmeans/', label)


def icaem(data, icVal, label, metric):
    ica_transformed_data = ica.return_independent_components(data, icVal)
    print('PCA transformed data shape  for em ', ica_transformed_data.shape)
    estimate_em_k({'features': ica_transformed_data}, label, 'plots/dr_clustering/ica/em/', metric)
    validate_em_k({'features': ica_transformed_data, 'labels': data['labels']}, 'plots/dr_clustering/ica/em/', label)

def rpkMeans(data,comopnents,label, metric):
    rp_transformed_data = randomprojection.return_random_components(data, comopnents)
    print('RP transformed data shape for kmeans ', rp_transformed_data.shape)
    estimate_k({'features': rp_transformed_data}, label, 'plots/dr_clustering/rp/kmeans/', metric)
    validate_k({'features': rp_transformed_data, 'labels': data['labels']}, 'plots/dr_clustering/rp/kmeans/', label)


def rpem(data, components, label, metric):
    rp_transformed_data = randomprojection.return_random_components(data, components)
    print('RP transformed data shape for kmeans ', rp_transformed_data.shape)
    estimate_em_k({'features': rp_transformed_data}, label, 'plots/dr_clustering/rp/em/', metric)
    validate_em_k({'features': rp_transformed_data, 'labels': data['labels']}, 'plots/dr_clustering/rp/em/', label)

def uvfskMeans(data,components,label, metric):
    uvfs_transformed_data = uvfs.apply_uvfs(data, label,components)
    print('uvfs transformed data shape for kmeans ', uvfs_transformed_data.shape)
    estimate_k({'features': uvfs_transformed_data}, label, 'plots/dr_clustering/uvfs/kmeans/', metric)
    validate_k({'features': uvfs_transformed_data, 'labels': data['labels']}, 'plots/dr_clustering/uvfs/kmeans/', label)


def uvfsem(data, components, label, metric):
    uvfs_transformed_data = uvfs.apply_uvfs(data, label, components)
    print('uvfs transformed data shape for kmeans ', uvfs_transformed_data.shape)
    estimate_em_k({'features': uvfs_transformed_data}, label, 'plots/dr_clustering/uvfs/em/', metric)
    validate_em_k({'features': uvfs_transformed_data, 'labels': data['labels']}, 'plots/dr_clustering/uvfs/em/', label)


def run_dr_clustering():
    deposit_data = load_cleanse_data()
    income_data = loadData()
    pcakMeans(deposit_data, 7, 'deposit', 'manhattan')
    # running PCA/kmeans for income Data
    pcakMeans(income_data, 4, 'income', 'euclidean')
    # running PCA/em for Deposit Data
    pcaem(deposit_data, 7, 'deposit', 'manhattan')
    # running PCA/em for Deposit Data
    pcaem(income_data, 4, 'income', 'euclidean')
    # ICA for deposit clustering
    icakMeans(deposit_data, 35, 'deposit', 'manhattan')
    icaem(deposit_data, 35, 'deposit', 'manhattan')
    # ICA for income clustering
    icakMeans(income_data, 12, 'income', 'euclidean')
    icaem(income_data, 12, 'income', 'euclidean')
    rpkMeans(deposit_data, 30,'deposit', 'manhattan')
    rpem(deposit_data, 30,'deposit', 'manhattan')
    # ICA for income clustering
    rpkMeans(income_data, 8,'income', 'euclidean')
    rpem(income_data, 8,'income', 'euclidean')
    uvfskMeans(deposit_data, 30,'deposit', 'manhattan')
    uvfsem(deposit_data, 30,'deposit', 'manhattan')
    uvfskMeans(income_data, 10, 'income', 'euclidean')
    uvfsem(income_data, 10, 'income', 'euclidean')

