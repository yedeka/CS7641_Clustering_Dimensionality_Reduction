import matplotlib.pyplot as plt
import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

def estimate_k(data, label, basePath, distance_metric):

    silhoute_score = []
    db_score = []
    run_times = []
    for k in range(2, 31):
        kmeans = KMeans(n_clusters=k, random_state=42)
        startTime = time.time()
        kmeans.fit(data['features'])
        endTime = time.time()
        silhoute_score.append(silhouette_score(data['features'], kmeans.labels_,metric=distance_metric))
        db_score.append(davies_bouldin_score(data['features'], kmeans.labels_))
        run_times.append(endTime - startTime)

    plt.style.use("fivethirtyeight")
    plt.plot(range(2, 31), silhoute_score)
    plt.xticks(range(2, 31), rotation="90")
    plt.xlabel("Number of Clusters")
    plt.ylabel('Silhoute Coefficients')
    plt.savefig(basePath + label + '/Silhouette_Coefficient')

    plt.clf()

    plt.style.use("fivethirtyeight")
    plt.plot(range(2, 31), db_score)
    plt.xticks(range(2, 31), rotation="90")
    plt.xlabel("Number of Clusters")
    plt.ylabel('Davies Bouldin Score')
    plt.savefig(basePath + label + '/DB_Score')
    plt.clf()

    plt.style.use("fivethirtyeight")
    plt.plot(range(2, 31), run_times)
    plt.xticks(range(2, 31), rotation="90")
    plt.xlabel("Number of Clusters")
    plt.ylabel('Run Times')
    plt.savefig(basePath + label + '/Run_Times')
    plt.clf()

def validate_k(data, basePath, label):
    features = data['features']
    rand_score = []

    for k in range(2, 31):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit_predict(features)
        labels = data['labels']
        rand_score.append(adjusted_rand_score(labels, kmeans.labels_))

    plt.style.use("fivethirtyeight")
    plt.plot(range(2, 31), rand_score)
    plt.xticks(range(2, 31), rotation="90")
    plt.xlabel("Number of Clusters")
    plt.ylabel('Adjusted Rand Score')
    plt.savefig(basePath + label + '/Adj_Rand_Score')
    plt.clf()
