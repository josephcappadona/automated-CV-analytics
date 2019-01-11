from math import sqrt, log
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


def get_optimal_clustering(data_to_cluster, cluster_sizes=[1,2,4,8,16,32,64,128,256,512,1024], show=True):
    if len(cluster_sizes) > 1:
        print('Finding optimal k...')
    else:
        print('Finding clusters...')
        
    models = {}
    scores = []
    for k in cluster_sizes:
        print('k: %s' % k, end='', flush=True)
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data_to_cluster)
        score = log(kmeans.inertia_)
        print(', score: %f' % score)
        models[k] = kmeans
        scores.append(score)
    cluster_score_data = np.stack((cluster_sizes, scores), axis=-1)
    k = get_knee_point(cluster_score_data, show=show)

    if len(cluster_sizes) > 1:
        print('Optimal k: %d' % k)
    else:
        print('Clusters found.')
    return models[k], k


def get_knee_point(data, show=True):
    # adapted from https://stackoverflow.com/questions/2018178/finding-the-best-trade-off-point-on-a-curve
    
    n_points = data.shape[0]
    first_point = data[0,:]
    last_point = data[-1,:]
    line_vec = last_point - first_point
    line_vec_normed = line_vec / sqrt(line_vec.dot(line_vec))
    vec_from_first = [point - first_point for point in data]
    line_vec_normed_many = np.repeat(np.array([line_vec_normed]), n_points, axis=0)
    scalar_product = (vec_from_first * line_vec_normed_many).sum(axis=1)
    vec_from_first_parallel = line_vec_normed_many* scalar_product[:,np.newaxis]
    vec_to_line = vec_from_first - vec_from_first_parallel
    
    vfunc_sqrt = np.vectorize(sqrt)
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
    