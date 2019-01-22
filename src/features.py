import numpy as np


def extract_features(im, cluster_model, descriptor_extractor, n_bins=3, mask=None):
    
    h, w = im.shape[:2]
    cluster_matrix = -np.ones((h, w), dtype=np.int16)
    bovw_histogram = np.zeros((cluster_model.n_clusters), dtype=np.uint8)
    
    keypoints, descriptors = descriptor_extractor.detectAndCompute(im, mask=mask)
    clusters = cluster_model.predict(descriptors)
    for kp, des, c_i in zip(keypoints, descriptors, clusters):
        x, y = [int(round(coord)) for coord in kp.pt]
        cluster_matrix[y, x] = c_i
        bovw_histogram[c_i] += 1
        
    return bovw_histogram, cluster_matrix
