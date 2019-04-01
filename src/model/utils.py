import time
import cv2
import numpy as np
from features import extract_features
import os
import logging

logging.getLogger('matplotlib').setLevel(logging.WARNING)


class Stopwatch(object):
    
    def __init__(self):
        self.start_ = None
        self.end = None
        self.duration = 0
    
    def start(self):
        self.start_ = time.time()
        
    def stop(self):
        self.end = time.time()
        self.duration += self.end - self.start_
    
    def format_str(self):
        m = int(round(self.duration / 60)); m_str = ('%dm' % m) if m > 0 else ''
        s = int(round(self.duration % 60)); s_str = '%ds' % s
        return m_str + s_str

def get_directory(filepath):
    return '/'.join(filepath.split('/')[:-1])

def get_filename(filepath):
    return filepath.split('/')[-1]

def get_parent_folder(filepath):
    return filepath.split('/')[-2]

def remove_extension(filename):
    return filename[:filename.rfind('.')]

def import_images(im_fps):
    logging.debug('Importing images...\nTotal images: %d' % len(im_fps))

    sw = Stopwatch(); sw.start()
    ims = []
    n = int(len(im_fps) / 10)
    for i, im_fp in enumerate(im_fps):
        if i % n == 0:
            logging.debug(i)
        ims.append(cv2.imread(im_fp))
    sw.stop()
    
    logging.debug('Import took %s.' % sw.format_str())
    return ims


def preprocess_images(ims, gaussian_kernel_radius=None):
    logging.debug('Preprocessing %d images...' % len(ims))
    sw = Stopwatch(); sw.start()

    new_ims = []
    for im in ims:
        if gaussian_kernel_radius:
            kernel = (gaussian_kernel_radius, gaussian_kernel_radius)
            im = cv2.GaussianBlur(im, kernel, cv2.BORDER_DEFAULT)
        new_ims.append(im)

    sw.stop()
    logging.debug('Done preprocessing images. Took %s.' % sw.format_str())
    return new_ims

def create_descriptor_extractor(de_type, de_params):
    if de_type == 'ORB':
        return cv2.ORB_create(10**7, **de_params)
    else:
        raise ValueError('Unsupported descriptor extractor type \'%s\'.' % de_type)

def get_descriptors(ims, descriptor_extractor):
    logging.debug('Processing image descriptors...')
    sw = Stopwatch(); sw.start()
    
    descriptors = []
    n = int(len(ims) / 10)
    for i, im in enumerate(ims):
        if i % n == 0:
            logging.debug(i)
        kp, des = descriptor_extractor.detectAndCompute(im, None)
        try:
            descriptors.extend(des)
        except TypeError:
            kp, des = kp_and_des_for_blank_image(im, descriptor_extractor)
            descriptors.extend(des)
        
    sw.stop()
    logging.debug('Total number of descriptors: %d' % len(descriptors))
    logging.debug('Done processing image descriptors. Took %s.' % sw.format_str())
    return descriptors

from cv2 import KeyPoint
def kp_and_des_for_blank_image(im, descriptor_extractor):
    h, w = im.shape[:2]
    x, y = (int(w/2), int(h/2))
    size = 1
    kp = KeyPoint(x, y, size)
    des = np.zeros((descriptor_extractor.descriptorSize()), dtype=np.uint8)
    return [kp], [des]
    
def get_histograms(ims, BOVW, descriptor_extractor, consider_colors, n_bins_per_channel=4):

    features_string = 'BOVW' + ('+colors' if consider_colors else '')
    logging.debug('Making %d %s histograms...' % (len(ims), features_string))
    sw = Stopwatch(); sw.start()

    histograms = []
    for im in ims:

        (bovw_histogram, _), (color_histogram, _) = \
            extract_features(im,
                             BOVW,
                             descriptor_extractor,
                             consider_colors,
                             n_bins_per_channel=n_bins_per_channel)

        if consider_colors:
            complex_histogram = np.hstack((bovw_histogram, color_histogram))
            histograms.append(complex_histogram)
        else:
            histograms.append(bovw_histogram)
    vstacked = np.vstack(histograms)

    sw.stop()
    logging.debug('Done making histograms. Took %s.' % sw.format_str())
    return vstacked

def get_params_string(params):
    return ', '.join('%s=%s' % (k, v) for k,v in params.items())
    
def train(classifier, X, y):
    logging.debug('Fitting model...')
    sw = Stopwatch(); sw.start()
    
    classifier.fit(X, y)
    
    sw.stop()
    logging.debug('Done fitting model. Took %s.' % sw.format_str())
    

def get_labels_from_fps(im_fps):
    text_labels = np.array([im_fp.split('/')[-1].split('.')[0] for im_fp in im_fps]) # .../FD.6.png -> FD
    return text_labels
  

def compute_error(im_labels, predictions):
    num_incorrect = sum(1 if l != p else 0 for (l, p) in zip(im_labels, predictions))
    num_total = len(im_labels)
    return num_incorrect / num_total

def get_score(Y, Y_hat):
    if len(Y) != len(Y_hat):
        raise ValueError
    return sum([1 if y == y_hat else 0 for y, y_hat in zip(Y, Y_hat)]) / len(Y)


def save_model(model, model_output_fp):
    model_output_dir = get_directory(model_output_fp)
    os.makedirs(model_output_dir, exist_ok=True)
    model.save(model_output_fp)

