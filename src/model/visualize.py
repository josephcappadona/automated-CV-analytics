from sklearn.decomposition import PCA
import colorsys
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import random
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import cv2
import utils
import os
from collections import defaultdict


# high level function used for making visualizations by passing in a command line arg
def create_visualizations(all_models, visualize_params, save_vis, model_output_fp):

    figs = set([o.lower() for o in visualize_params.split(',')])

    if 'bovw_pca' in figs:
        visualize_BOVW_PCA(all_models[0][0], save=save_vis, model_fp=model_output_fp)

    if 'bovw_examples' in figs:
        visualize_BOVW_examples(all_models[0][0], save=save_vis, model_fp=model_output_fp)
    
    if 'bovw_keypoints' in figs:
        visualize_BOVW_keypoints(all_models[0][0], save=save_vis, model_fp=model_output_fp)

    if 'n_train_examples' in figs:
        visualize_parameter(all_models, 'n_train_examples', save=save_vis, model_fp=model_output_fp)

    if 'n_clusters' in figs:
        visualize_parameter(all_models, ('cluster_model_params', 'n_clusters'), xscale='log', save=save_vis, model_fp=model_output_fp)

    if 'spatial_pyramid_levels' in figs:
        visualize_parameter(all_models, 'spatial_pyramid_levels', save=save_vis, model_fp=model_output_fp)

    if 'feature_selection_threshold' in figs:
        visualize_parameter(all_models, ('feature_selection_params', 'threshold'), save=save_vis, model_fp=model_output_fp)

    if 'descriptor_extractor_threshold' in figs:
        visualize_parameter(all_models, ('descriptor_extractor_params', 'threshold'), xscale='log', save=save_vis, model_fp=model_output_fp)


# visualizes BOVW clusters in 3 dimensions (using PCA)
def visualize_BOVW_PCA(model, save=False, show=True, model_fp=None):

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

    if save:
        fig_fp = get_fig_filepath('bovw_pca', model_fp)
        save_figure(fig, fig_fp)
    if show:
        plt.show()


# extracts visual word patches and displays them in their clusters
def visualize_BOVW_examples(model, save=False, show=True, model_fp=None):
   
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
    fig, axarr = plt.subplots(n_rows, n_cols)
    fig.suptitle('Bag of Visual Word samples', fontsize=20)
    fig.set_size_inches(2+n_cols, 0.5+n_rows)
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

    if save:
        fig_fp = get_fig_filepath('bovw_examples', model_fp)
        save_figure(fig, fig_fp)
    if show:
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
def visualize_BOVW_keypoints(model, save=False, show=True, model_fp=None):
    
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
    fig, axarr = plt.subplots(n_rows, n_cols)
    fig.suptitle('Snippet Keypoint Visualization', fontsize=20)
    fig.set_size_inches(2+n_cols*2, 2+n_rows*1.25)
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
        ax.set_title(label, pad=20, fontdict={'fontsize':8, 'fontweight':'semibold'})

    plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.1, wspace=0.4, hspace=0.4)

    if save:
        fig_fp = get_fig_filepath('bovw_keypoints', model_fp)
        save_figure(fig, fig_fp)
    if show:
        plt.show()


# graphs visualization of cross_val_err and test_acc with changes in a parameter
def visualize_parameter(results, parameter, save=False, show=True, model_fp=None, **kwargs):
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
    if 'xscale' in kwargs:
        ax.set_xscale(kwargs['xscale'])
        ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        #ax.xaxis.set_minor_formatter(FormatStrFormatter('%g'))

    # label points
    for x, (err, acc) in zip(x, zip(errs, accs)):
        ax.annotate('(%d, %g)' % (x, err), (x,err), xytext=(0,9), textcoords='offset points', ha='center')
        ax.annotate('(%d, %g)' % (x, acc), (x,acc), xytext=(0,-16), textcoords='offset points', ha='center')

    if save:
        fig_fp = get_fig_filepath(parameter, model_fp)
        save_figure(fig, fig_fp)
    if show:
        plt.show()


# methods for saving graphics

def save_figure(fig, fig_fp):
    fig.savefig(fig_fp)

def get_fig_filepath(parameter, model_fp):
    model_dir = get_directory(model_fp)
    model_fn_base = remove_extension(get_filename(model_fp))

    fig_fn = model_fn_base + '.' + parameter + '_vis.png'
    fig_fp = os.path.join(model_dir, fig_fn)
    return fig_fp

def get_filename(fp):
    return fp.split('/')[-1]

def remove_extension(fn):
    return ".".join(fn.split('.')[:-1])

def get_directory(fp):
    return "/".join(fp.split('/')[:-1])


# adapted from https://stackoverflow.com/a/47194111
def get_N_HexCol(N):
    HSV_tuples = [(x * 1.0 / N, 0.75 + y * 0.25 / N, 0.75 + z * 0.25 / N) for x,y,z in zip(range(N), sorted(list(range(N)), key=lambda x: random.random()), sorted(list(range(N)), key=lambda x: random.random()))]
    hex_out = []
    for rgb in HSV_tuples:
        rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
        hex_out.append('#%02x%02x%02x' % tuple(rgb))
    return hex_out

