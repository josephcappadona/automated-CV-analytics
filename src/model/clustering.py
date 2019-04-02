import math
import numpy as np
import matplotlib.pyplot as plt
from utils import Stopwatch
from sklearn.cluster import MiniBatchKMeans, KMeans, DBSCAN
import logging

def get_clustering(data_to_cluster, cluster_model_type, cluster_model_params):
    sw = Stopwatch(); sw.start()

    if cluster_model_type == 'KMEANS':
        cluster_model = MiniBatchKMeans(**cluster_model_params)
    elif cluster_model_type == 'DBSCAN':
        cluster_model = DBSCAN_w_prediction(**cluster_model_params)

    cluster_model.fit(data_to_cluster)

    sw.stop()
    logging.debug('Descriptors clustered into %d clusters.' % cluster_model.n_clusters)
    return cluster_model


def get_optimal_clustering(data_to_cluster, cluster_sizes=[2,4,8,16,32,64,128,256,512,1024], show=True):
    logging.debug('Finding optimal k...')
        
    models = {}
    scores = []
    for k in cluster_sizes:
        models[k] = get_clustering(data_to_cluster, k)
        scores.append(math.log(models[k].inertia_))

    # stack data for graphing
    cluster_score_data = np.stack((cluster_sizes, scores), axis=-1)

    # use elbow method to find optimal k
    optimal_k = get_knee_point(cluster_score_data, show=show)
    logging.debug('Optimal k: %d' % optimal_k)

    return models[optimal_k]


def get_knee_point(data, show=True):
    # Uses elbow method to find optimal value of k
    # adapted from https://stackoverflow.com/questions/2018178/finding-the-best-trade-off-point-on-a-curve
    # "A quick way of finding the elbow is to draw a line from the
    #  first to the last point of the curve and then find the data
    #  point that is farthest away from that line."
    
    n_points = data.shape[0]
    first_point = data[0,:]
    last_point = data[-1,:]
    line_vec = last_point - first_point
    line_vec_normed = line_vec / math.sqrt(line_vec.dot(line_vec))
    vec_from_first = [point - first_point for point in data]
    line_vec_normed_many = np.repeat(np.array([line_vec_normed]), n_points, axis=0)
    scalar_product = (vec_from_first * line_vec_normed_many).sum(axis=1)
    vec_from_first_parallel = line_vec_normed_many* scalar_product[:,np.newaxis]
    vec_to_line = vec_from_first - vec_from_first_parallel
    
    vfunc_sqrt = np.vectorize(math.sqrt)
    dist_to_line = vfunc_sqrt((vec_to_line * vec_to_line).sum(axis=1))
    
    if show:
        plt.figure()
        plt.plot(data[:,0], data[:,1], 'bx-')
        plt.xlabel('k')
        plt.ylabel('sum of squared distances')
        plt.title('Elbow Method For Optimal k')

        plt.figure()
        plt.plot(data[:,0], dist_to_line, 'bx-')
        plt.xlabel('k')
        plt.ylabel('distance to line')
        plt.title('Knee Point For Optimal k')
        plt.show()
    
    index_of_max = max(enumerate(dist_to_line), key=lambda i_dist: i_dist[1])[0]
    k = data[index_of_max, 0]
    return k

class DBSCAN_w_prediction(DBSCAN):

    def fit(self, *args, **kwargs):
        ret = super(DBSCAN_A, self).fit(*args, **kwargs)
        self.n_clusters = len(set(self.labels_))
        print('\n')
        print(self.n_clusters)
        print('\n')
        return ret

    " taken from https://stackoverflow.com/a/51516334"
    def predict(self, X):

        nr_samples = X.shape[0]

        y_new = np.ones(shape=nr_samples, dtype=int) * self.n_clusters

        for i in range(nr_samples):
            diff = self.components_ - X[i, :]  # NumPy broadcasting

            dist = np.linalg.norm(diff, axis=1)  # Euclidean distance

            shortest_dist_idx = np.argmin(dist)

            if dist[shortest_dist_idx] < self.eps:
                y_new[i] = self.labels_[self.core_sample_indices_[shortest_dist_idx]]

        return y_new
       
