from sklearn.decomposition import PCA
import colorsys
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import cv2
import utils
from collections import defaultdict


# visualizes BOVW clusters in 3 dimensions (using PCA)
def visualize_BOVW_PCA(model):

    BOVW = model.BOVW
    descriptors = BOVW.X
    clusters = BOVW.predict(descriptors)

    reduced_descriptors = PCA(n_components=3).fit_transform(descriptors)
    label_to_color = get_N_HexCol(BOVW.n_clusters)

    fig = plt.figure()
    fig.suptitle('Bag of Visual Words clusters (PCA)', fontsize=20)
    ax = Axes3D(fig)
    for c_i, d_i, pca_v in zip(clusters, descriptors, reduced_descriptors):

        ax.scatter(*pca_v, c=label_to_color[c_i])

    plt.show()


# extracts visual word patches and displays them in their clusters
def visualize_BOVW_samples(model):
   
    BOVW = model.BOVW
    keypoints = BOVW.kp
    clusters = BOVW.clusters
    ims = BOVW.ims

    d = defaultdict(list)
    for im, kps, cs in zip(ims, keypoints, clusters):
        for kp, c in zip(kps, cs):
            (x_min, x_max), (y_min, y_max) = get_kp_bbox(kp)
            kp_im = Image.fromarray(im[y_min:y_max, x_min:x_max])
            d[c].append(kp_im)
    
    n_rows = n_cols = 8
    f, axarr = plt.subplots(n_rows, n_cols)
    f.suptitle('Bag of Visual Word samples', fontsize=20)
    f.set_size_inches(2+n_cols, 0.5+n_rows)
    for r in range(n_rows):
        for c in range(n_cols):
            try:
                VW_im = d[c][r]
                axarr[r, c].imshow(VW_im)
                axarr[r, c].tick_params(labelsize=8, which='major')
                axarr[r, c].minorticks_off()
            except (IndexError, KeyError):
                pass
    # label columns
    for c, ax in enumerate(axarr[0]):
        ax.set_title('cluster #%d' % (c+1), pad=20, fontdict={'fontsize':8, 'fontweight':'semibold'})

    plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.05, wspace=1, hspace=0.5)
    plt.show()

def get_kp_bbox(kp):
    x, y = kp.pt
    r = kp.size / 2
    x_min = int(x - r)
    x_max = int(x + r)
    y_min = int(y - r)
    y_max = int(y + r)
    return (x_min, x_max), (y_min, y_max)


# draws keypoints over small sample of positive and negative images
def visualize_keypoints(model):
    
    BOVW = model.BOVW
    keypoints = BOVW.kp
    ims = BOVW.ims
    im_labels = BOVW.im_labels

    d = defaultdict(list)
    sorted_labels = sorted(set(im_labels))
    for im, im_label, kps in zip(ims, im_labels, keypoints):
        cv2_kps = utils.convert_custom_kps_to_cv2_kps(kps)
        kp_im = Image.fromarray(cv2.drawKeypoints(im, cv2_kps, None, color=(0,255,0), flags=4))
        d[im_label].append(kp_im)

    n_rows, n_cols = 5, len(sorted_labels)
    f, axarr = plt.subplots(n_rows, n_cols)
    f.suptitle('Snippet Keypoint Visualization', fontsize=20)
    f.set_size_inches(2+n_cols*2, 2+n_rows*1.25)
    for r in range(n_rows):
        for c in range(n_cols):
            try:
                label = sorted_labels[c]
                kp_im = d[label][r]
                axarr[r, c].imshow(kp_im)
            except:
                pass # Not enough samples with this label
                
    # label columns
    for ax, label in zip(axarr[0], sorted_labels):
        ax.set_title(label, pad=10, fontdict={'fontsize':8, 'fontweight':'semibold'})

    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.1, wspace=0.4, hspace=0.4)
    plt.show()


# graphs visualization of cross_val_err and test_acc with changes in a parameter
def visualize_parameter(results, parameter):
    x = []
    errs = []
    accs = []
    for _, config, val_err, test_acc in results:
        if type(parameter) == tuple:
            arg, subarg = parameter
            x.append(config[arg][subarg])
        else:
            x.append(config[parameter])
        errs.append(val_err)
        accs.append(test_acc)
    
    fig, ax = plt.subplots()
    ax.plot(x, accs, marker='^', color='red', label='test_acc')
    ax.plot(x, errs, marker='s', color='blue', label='cross_val_err')
    ax.legend()

    parameter = parameter[1] if type(parameter) == tuple else parameter
    ax.set_title('Cross-Val Error, Test Accuracy vs %s' % parameter, pad=15)

    ax.set_xticks(x)
    ax.set_yticks([0, 1], minor=True)
    ax.set_xlabel(parameter)

    # label points
    for x, (err, acc) in zip(x, zip(errs, accs)):
        ax.annotate(' %g' % err, xy=(x,err), xycoords='data')
        ax.annotate(' %g' % acc, xy=(x,acc), xycoords='data')

    plt.show()


# adapted from https://stackoverflow.com/a/47194111
def get_N_HexCol(N):
    HSV_tuples = [(x * 1.0 / N, 0.75 + y * 0.25 / N, 0.75 + z * 0.25 / N) for x,y,z in zip(range(N), sorted(list(range(N)), key=lambda x: random.random()), sorted(list(range(N)), key=lambda x: random.random()))]
    hex_out = []
    for rgb in HSV_tuples:
        rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
        hex_out.append('#%02x%02x%02x' % tuple(rgb))
    return hex_out
