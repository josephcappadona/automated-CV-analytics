from sklearn.decomposition import PCA
import colorsys
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import cv2
import utils

def visualize_BOVW_PCA(model, plot_type='3D'):

    BOVW = model.BOVW
    descriptors = BOVW.X
    clusters = BOVW.predict(descriptors)

    reduced_descriptors = PCA(n_components=3).fit_transform(descriptors)
    label_to_color = get_N_HexCol(BOVW.n_clusters)

    if plot_type == '3D':
        fig = plt.figure()
        ax = Axes3D(fig)
    else:
        fig, axarr = plt.subplots(1, 3)
        fig.set_size_inches(12, 4)
        ax1, ax2, ax3 = axarr

    for c_i, d_i, pca_v in zip(clusters, descriptors, reduced_descriptors):

        col = label_to_color[c_i]
        if plot_type == '3D':
            ax.scatter(*pca_v, c=col)
        else:
            pca1, pca2, pca3 = pca_v
            ax1.scatter(pca1, pca2, c=col)
            ax2.scatter(pca1, pca3, c=col)
            ax3.scatter(pca2, pca3, c=col)

    plt.show()


def visualize_BOVW_samples(model):
   
    BOVW = model.BOVW
    keypoints = BOVW.kp
    ims = BOVW.ims

    id_ = 2

    im = ims[id_]; print(im.shape)
    kps = keypoints[id_]
    for k in kps:
        print(k.size, k.angle, k.class_id)
    kp = keypoints[id_][-1]
    (x_min, x_max), (y_min, y_max) = bbox = get_kp_bbox(kp)
    print(kp.pt, kp.size)
    print(bbox)
    im_o = Image.fromarray(im)
    im_k = Image.fromarray(im[y_min:y_max, x_min:x_max])
    kps_a = utils.convert_custom_kps_to_cv2_kps(kps)
    im_ks = Image.fromarray(cv2.drawKeypoints(im, kps_a, None, color=(0,255,0), flags=4))

    
    f, axarr = plt.subplots(3)
    f.set_size_inches(10, 4)
    axarr[0].imshow(im_o)
    axarr[1].imshow(im_k)
    axarr[2].imshow(im_ks)
    plt.show()

def get_kp_bbox(kp):
    x, y = kp.pt
    r = kp.size / 2
    x_min = int(round(x - r))
    x_max = int(round(x + r))
    y_min = int(round(y - r))
    y_max = int(round(y + r))
    return (x_min, x_max), (y_min, y_max)

" adapted from https://stackoverflow.com/a/47194111"
def get_N_HexCol(N):
    HSV_tuples = [(x * 1.0 / N, 0.75 + y * 0.25 / N, 0.75 + z * 0.25 / N) for x,y,z in zip(range(N), sorted(list(range(N)), key=lambda x: random.random()), sorted(list(range(N)), key=lambda x: random.random()))]
    hex_out = []
    for rgb in HSV_tuples:
        rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
        hex_out.append('#%02x%02x%02x' % tuple(rgb))
    return hex_out
