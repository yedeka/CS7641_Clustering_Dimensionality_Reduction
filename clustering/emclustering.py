import time
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score

def estimate_em_k(data, label, basepath, distance_metric):
    silhoute_score = []
    db_score = []
    run_times = []
    for k in range(2, 31):
        em = GMM(n_components=k, random_state=50)
        startTime = time.time()
        predicted_labels = em.fit_predict(data['features'])
        endTime = time.time()
        silhoute_score.append(silhouette_score(data['features'], predicted_labels, metric=distance_metric))
        db_score.append(davies_bouldin_score(data['features'], predicted_labels))
        run_times.append(endTime - startTime)

    plt.style.use("seaborn")
    plt.plot(range(2, 31), silhoute_score)
    plt.xticks(range(2, 31), rotation="90")
    plt.xlabel("Number of Clusters")
    plt.ylabel('Silhoute Coefficients')
    plt.savefig(basepath + label + '/Silhoute_Score')
    #plt.savefig('plots/em/' + label + '/Silhoute_Score')
    plt.clf()

    plt.style.use("seaborn")
    plt.plot(range(2, 31), db_score)
    plt.xticks(range(2, 31), rotation="90")
    plt.xlabel("Number of Clusters")
    plt.ylabel('David Bouldin Score')
    plt.savefig(basepath + label + '/Db_Score')
    plt.clf()

    plt.style.use("seaborn")
    plt.plot(range(2, 31), run_times)
    plt.xticks(range(2, 31), rotation="90")
    plt.xlabel("Number of Clusters")
    plt.ylabel('Run time')
    plt.savefig(basepath + label + '/Run_Time')
    plt.clf()

def validate_em_k(data, basepath, label):
    features = data['features']
    rand_score = []
    for k in range(2, 31):
        em = GMM(n_components=k, random_state=50)
        pred_labels = em.fit_predict(features)
        labels = data['labels']
        rand_score.append(adjusted_rand_score(labels, pred_labels))

    plt.style.use("seaborn")
    plt.plot(range(2, 31), rand_score)
    plt.xticks(range(2, 31), rotation="90")
    plt.xlabel("Number of Clusters")
    plt.ylabel('Adjusted Rand Score')
    plt.savefig(basepath + label + '/Adj_Rand_Score')
    plt.clf()

def validate_em_k_fixed(features, labels,k):
    features = features
    em = GMM(n_components=k, random_state=50)
    pred_labels = em.fit_predict(features)
    return adjusted_rand_score(labels, pred_labels)
