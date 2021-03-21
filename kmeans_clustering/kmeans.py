import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def estimate_k(data):
    kmeans_kwargs = {
        "init": "random",
        "n_init": 20,
        "max_iter": 500,
        "random_state": 42,
    }
    sse = []
    for k in range(1, 31):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(data['features'])
        sse.append(kmeans.inertia_)

    plt.style.use("fivethirtyeight")
    plt.plot(range(1, 31), sse)
    plt.xticks(range(1, 31),rotation=90)
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.show()

    silhouette_coefficients = []
    for k in range(2, 31):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(data['features'])
        score = silhouette_score(data['features'], kmeans.labels_)
        silhouette_coefficients.append(score)

    plt.plot(range(2, 31), silhouette_coefficients)
    plt.xticks(range(2, 31),rotation=90)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")
    plt.show()