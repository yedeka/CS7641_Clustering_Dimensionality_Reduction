from data.deposit_data_loader import load_cleanse_data
from data.income_evaluation_loader import loadData
from clustering.kmeans import estimate_k, validate_k
from clustering.emclustering import estimate_em_k, validate_em_k

def deposit_clustering():
    data = load_cleanse_data()
    estimate_k(data, 'deposit', 'plots/kmeans/', 'manhattan')
    validate_k(data, 'plots/kmeans/', 'deposit')

def income_clustering():
    data = loadData()
    estimate_k(data, 'income', 'plots/kmeans/', 'euclidean')
    validate_k(data, 'plots/kmeans/', 'income')

def deposit_em():
    data = load_cleanse_data()
    estimate_em_k(data, 'deposit', 'plots/em/', 'manhattan')
    validate_em_k(data, 'plots/em/', 'deposit')

def income_em():
    data = loadData()
    print(data)
    estimate_em_k(data, 'income', 'plots/em/', 'euclidean')
    validate_em_k(data, 'plots/em/', 'income')

def clusteringExpt():
    deposit_clustering()
    income_clustering()
    deposit_em()
    income_em()
