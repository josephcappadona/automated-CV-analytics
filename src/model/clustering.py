import math
import numpy as np
import matplotlib.pyplot as plt
from utils import Stopwatch
from sklearn.cluster import MiniBatchKMeans
import logging

def get_clustering(data_to_cluster, k):
    sw = Stopwatch(); sw.start()

    kmeans = MiniBatchKMeans(n_clusters=k)
    kmeans.fit(data_to_cluster)
    score = math.log(kmeans.inertia_)

    sw.stop()
    logging.debug('k: %s, log(inertia): %.5f, time: %s' % (k, score, sw.format_str()))
    return kmeans


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
    
