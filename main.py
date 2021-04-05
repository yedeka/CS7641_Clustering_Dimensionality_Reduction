from clustering import performclustering
from dr import performDr
from drclustering import performdrclustering
from clusteringNN import performKmeansNN

if __name__ == '__main__':
    performclustering.clusteringExpt()
    performDr.performDR()
    performdrclustering.run_dr_clustering()
    performKmeansNN()
