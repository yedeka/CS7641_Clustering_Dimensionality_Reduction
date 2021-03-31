import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from kneed import KneeLocator
from yellowbrick.cluster import InterclusterDistance

def estimate_k(data, label, distance_metric, useIntercluster):
    kmeans_kwargs = {
        "random_state": 42,
    }
    # Try to determine K using elbow
    sse = []
    for k in range(2, 31):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(data['features'])
        sse.append(kmeans.inertia_)

    kl = KneeLocator(
        range(2, 31), sse, curve="convex", direction="decreasing"
    )
    print(kl.elbow)
    print(kl.knee)
    plt.style.use('ggplot')
    kl.plot_knee()
    plt.savefig('../plots/'+label+'_elbow.png')
    plt.clf()

    # Since elbow does not give us a clear distinction try to do with Silhoute score
    silhoute_score = []
    for k in range(2, 31):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(data['features'])
        silhoute_score.append(silhouette_score(data['features'], kmeans.labels_,metric=distance_metric))

    plt.style.use("fivethirtyeight")
    plt.plot(range(2, 31), silhoute_score)
    plt.xticks(range(2, 31), rotation="90")
    plt.xlabel("Number of Clusters")
    plt.ylabel('Silhoute Coefficients')
    plt.savefig('../plots/'+label+"_Silhouette Coefficient")
    plt.clf()

    if useIntercluster:
        for k in range(2, 10):
            model = KMeans(k)
            visualizer = InterclusterDistance(model)
            visualizer.fit(data['features'])
            visualizer.show()
            plt.clf()

    db_score = []
    for k in range(2, 31):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(data['features'])
        db_score.append(davies_bouldin_score(data['features'], kmeans.labels_))

    plt.style.use("fivethirtyeight")
    plt.plot(range(2, 31), db_score)
    plt.xticks(range(2, 31), rotation="90")
    plt.xlabel("Number of Clusters")
    plt.ylabel('Davies Bouldin Score')
    plt.savefig('../plots/' + label + "_DB Score")
    plt.clf()

def validate_k(k,data):
    features = data['features']
    kmeans = KMeans(n_clusters=k)
    kmeans.fit_predict(features)
    labels = data['labels']
    rand_score = adjusted_rand_score(labels, kmeans.labels_)
    return rand_score

def apply_kmeans(data, k):
    model = KMeans(k)
    visualizer = InterclusterDistance(model)
    visualizer.fit(data['features'])  # Fit the data to the visualizer
    visualizer.show()
