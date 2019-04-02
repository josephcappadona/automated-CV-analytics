import numpy as np
import utils
import cv2


def build_histogram(im, cluster_model, keypoints, descriptors, spatial_pyramid_levels, n_bins_per_channel=4):
    
    h, w = im.shape[:2]

    bovw_size = cluster_model.n_clusters
    n_color_channels = 1 if len(im.shape) == 2 else im.shape[2]
    n_colors = n_bins_per_channel ** n_color_channels

    cluster_matrix = -np.ones((h, w), dtype=np.int16)
    color_matrix = -np.ones((h, w, n_color_channels), dtype=np.int8)

    # build empty pyramid
    histogram_pyramid = []
    for l in range(spatial_pyramid_levels):
        l_i_histogram = np.zeros((2**l, 2**l, bovw_size+n_colors))
        histogram_pyramid.append(l_i_histogram)

    # iterate over keypoints+descriptors,
    # record each descriptor's cluster id and color in relevant data structures
    cluster_predictions = cluster_model.predict(descriptors)
    for kp, des, c_i in zip(keypoints, descriptors, cluster_predictions):

        x, y = [int(round(coord)) for coord in kp.pt]
        color_bin = get_bin_for_color(im[y, x], n_color_channels, n_bins_per_channel)

        cluster_matrix[y, x] = c_i
        color_matrix[y, x] = color_bin

        for l in range(spatial_pyramid_levels):
            l_x, l_y = int(x / (w/2**l)), int(y / (h/2**l))
            histogram_pyramid[l][l_x, l_y, c_i] += 1

    # concatenate pyramid levels into single histogram
    for l in range(spatial_pyramid_levels):
        histogram_pyramid[l] = np.hstack(np.hstack(histogram_pyramid[l]))
    full_histogram = np.hstack(histogram_pyramid)
    return full_histogram, cluster_matrix, color_matrix

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


def extract_features(im, cluster_model, descriptor_extractor, spatial_pyramid_levels, n_bins_per_channel=4):

    # extract keypoints and descriptors
    keypoints, descriptors = descriptor_extractor.detectAndCompute(im, None)
    if not len(keypoints):
        keypoints, descriptors = utils.kp_and_des_for_blank_image(im, descriptor_extractor)
    descriptors = np.array(descriptors)
    
    # extract BOVW info
    full_histogram, cluster_matrix, color_matrix = build_histogram(im, cluster_model, keypoints, descriptors, spatial_pyramid_levels)

    return full_histogram, cluster_matrix, color_matrix

