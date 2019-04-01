import numpy as np
import utils
import cv2


def extract_bovw_info(im, cluster_model, keypoints, descriptors):
    
    h, w = im.shape[:2]

    cluster_matrix = -np.ones((h, w), dtype=np.int16)
    bovw_histogram = np.zeros((cluster_model.n_clusters), dtype=np.uint8)
    l_1_histograms = np.zeros((2, 2, cluster_model.n_clusters), dtype=np.uint8)
    l_2_histograms = np.zeros((4, 4, cluster_model.n_clusters), dtype=np.uint8)
    
    clusters = cluster_model.predict(descriptors)
    # iterate over keypoints+descriptors,
    # record each descriptor's cluster id in `cluster_matrix` and `cluster_histogram`
    for kp, des, c_i in zip(keypoints, descriptors, clusters):
        x, y = [int(round(coord)) for coord in kp.pt]
        cluster_matrix[y, x] = c_i
        bovw_histogram[c_i] += 1

        l_1_x, l_1_y = int(x / (w/2)), int(y / (h/2))
        l_1_histograms[l_1_x, l_1_y, c_i] += 1
        l_2_x, l_2_y = int(x / (w/4)), int(y / (h/4))
        l_2_histograms[l_2_x, l_2_y, c_i] += 1
       
    l_1_histogram = np.hstack(np.hstack(l_1_histograms))
    l_2_histogram = np.hstack(np.hstack(l_2_histograms))
    #bovw_histogram = np.hstack((bovw_histogram, l_1_histogram))
    return bovw_histogram, cluster_matrix


def get_bin_for_color(pixel, n_color_channels, n_bins_per_channel):
    if n_color_channels == 1: pixel = [pixel]

    bin_ = 0
    for i, channel in enumerate(pixel):
        reduced_channel = int(channel / (256 / n_bins_per_channel))
        bin_ += (n_bins_per_channel ** i) * reduced_channel

        # reduces the number of colors in each channel to `n_bins_per_channel`
        # then maps each color to a unique integer
        # number of bins = n_bins_per_channel ** n_color_channels
        #     e.g., 3 color channels, 4 bins per channel  =>  64 bins

        # if n_b_p_c=4, then 0-63=>0, 64-127=>1, 128-191=>2, 192-255=>3
        # so (50, 150, 250) => (0, 2, 3)
        # then, bin = 0*(4**0) + 2*(4**1) + 3*(4**2) = 56
    return bin_

def extract_color_info(im, keypoints, n_bins_per_channel=4):
    
    h, w = im.shape[:2]
    n_color_channels = 1 if len(im.shape) == 2 else im.shape[2]
    n_colors = n_bins_per_channel ** n_color_channels

    color_matrix = -np.ones((h, w, n_color_channels), dtype=np.int8)
    color_histogram = np.zeros((n_colors), dtype=np.uint8)
    
    # iterate over keypoints,
    # record color at each in `color_matrix` and `color_histogram`
    for kp in keypoints:
        x, y = [int(round(coord)) for coord in kp.pt]
        bin_ = get_bin_for_color(im[y, x], n_color_channels, n_bins_per_channel)
        color_matrix[y, x] = bin_
        color_histogram[bin_] += 1
        
    return color_histogram, color_matrix

def extract_features(im, cluster_model, descriptor_extractor, consider_colors, n_bins_per_channel=4):

    # extract keypoints and descriptors
    keypoints, descriptors = descriptor_extractor.detectAndCompute(im, None)
    if not len(keypoints):
        keypoints, descriptors = utils.kp_and_des_for_blank_image(im, descriptor_extractor)
    
    # extract BOVW info
    bovw_histogram, cluster_matrix = extract_bovw_info(im, cluster_model, keypoints, descriptors)

    # extract color info
    if consider_colors:
        color_histogram, color_matrix = extract_color_info(im, keypoints, n_bins_per_channel=n_bins_per_channel)
    else:
        color_histogram, color_matrix = [], []

    return ((bovw_histogram, cluster_matrix), (color_histogram, color_matrix))

