import numpy as np


def extract_bovw_info(im, cluster_model, keypoints, descriptors, n_bins_per_color=4, mask=None):
    
    h, w = im.shape[:2]

    cluster_matrix = -np.ones((h, w), dtype=np.int16)
    bovw_histogram = np.zeros((cluster_model.n_clusters), dtype=np.uint8)
    
    clusters = cluster_model.predict(descriptors)
    for kp, des, c_i in zip(keypoints, descriptors, clusters):
        x, y = [int(round(coord)) for coord in kp.pt]
        cluster_matrix[y, x] = c_i
        bovw_histogram[c_i] += 1
        
    return bovw_histogram, cluster_matrix


def get_bin_for_color(pixel, n_color_channels, n_bins_per_color):
    if n_color_channels == 1: pixel = [pixel]

    bin_ = 0
    for i, channel in enumerate(pixel):
        reduced_channel = int(channel / (256 / n_bins_per_color))
        bin_ += (n_bins_per_color ** i) * reduced_channel
        # reduces the number of colors in each channel to `n_bins_per_colors`
        # then maps each color to a unique integer (essentially a conversion from base `n_bins_per_color` to base 10)

        # if n_b_p_c=4, then 0-63=>0, 64-127=>1, 128-191=>2, 192-255=>3
        # so (50, 150, 250) => (0, 2, 3)
        # then, bin = 0*(4**0) + 2*(4**1) + 3*(4**2)
    return bin_

def extract_color_info(im, keypoints, n_bins_per_color=4, mask=None):
    
    h, w = im.shape[:2]
    n_color_channels = 1 if len(im.shape) == 2 else im.shape[2]
    n_colors = n_bins_per_color ** n_color_channels

    color_matrix = -np.ones((h, w, n_color_channels), dtype=np.int8)
    color_histogram = np.zeros((n_colors), dtype=np.uint8)
    
    for kp in keypoints:
        x, y = [int(round(coord)) for coord in kp.pt]
        bin_ = get_bin_for_color(im[y, x], n_color_channels, n_bins_per_color)
        color_matrix[y, x] = bin_
        color_histogram[bin_] += 1
        
    return color_histogram, color_matrix


def extract_features(im, cluster_model, descriptor_extractor, n_bins_per_color=4, mask=None):

    keypoints, descriptors = descriptor_extractor.detectAndCompute(im, mask=mask)

    bovw_histogram, cluster_matrix = extract_bovw_info(im, cluster_model, keypoints, descriptors, mask=mask)
    color_histogram, color_matrix = extract_color_info(im, keypoints, n_bins_per_color=n_bins_per_color, mask=mask)

    return ((bovw_histogram, cluster_matrix), (color_histogram, color_matrix))
