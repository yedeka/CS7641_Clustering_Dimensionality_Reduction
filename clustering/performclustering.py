from data.deposit_data_loader import load_cleanse_data
from data.income_evaluation_loader import loadData
from clustering.kmeans import estimate_k, validate_k, apply_kmeans

def deposit_clustering():
    data = load_cleanse_data()
    # Since after discretizing total features are coming out to be 41 I will use Manhatten distance as it fits better for more number of features
    estimate_k(data, 'deposit','manhattan', False)
    # We will validate k obtained from elbow/Silhoute_score/Davies_Bouldin score
    elbow_rand_score = validate_k(18,data)
    print('deposit_elbow_rand_score',elbow_rand_score)
    sc_rand_score = validate_k(2, data)
    print('deposit_silhoute_rand_score', sc_rand_score)
    db_rand_score = validate_k(28, data)
    print('deposit_db_rand_score', db_rand_score)
    apply_kmeans(data,2)


def income_clustering():
    data = loadData()
    estimate_k(data, 'income','euclidean', True)
    # We will validate k obtained from elbow/Silhoute_score/Davies_Bouldin score
    elbow_rand_score = validate_k(10, data)
    '''print('income_elbow_rand_score', elbow_rand_score)
    sc_rand_score = validate_k(8, data)
    print('income_silhoute_rand_score', sc_rand_score)
    db_rand_score = validate_k(8, data)
    print('income_db_rand_score', db_rand_score)
    apply_kmeans(data, 8)'''

if __name__ == '__main__':
    #deposit_clustering()
    income_clustering()
