from data.deposit_data_loader import load_cleanse_data
from kmeans_clustering.kmeans import estimate_k

def deposit_clustering():
    data = load_cleanse_data()
    estimate_k(data)

if __name__ == '__main__':
    deposit_clustering()
