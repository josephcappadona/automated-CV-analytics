import time
import cv2
from cv2 import KeyPoint
import numpy as np
from features import extract_features
import os
from collections import defaultdict
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
    if not im_fps:
        logging.error('No images to import. Exiting...')
        exit()

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
    return np.array(ims)

def remove_extra_examples(train_ims, train_im_labels, n_train_samples):
    condensed_ims, condensed_im_labels = [], []
    count_dict = defaultdict(int)
    for im, label in zip(train_ims, train_im_labels):
        if label != 'NEGATIVE' and count_dict[label] >= n_train_samples:
            continue
        else:
            count_dict[label] += 1
            condensed_ims.append(im)
            condensed_im_labels.append(label)
    return np.array(condensed_ims), np.array(condensed_im_labels)


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
    logging.debug('Creating %s descriptor extractor with params \'%s\'.' % (de_type, get_params_string(de_params)))
    if de_type == 'ORB':
        return cv2.ORB_create(10**7, **de_params)
    elif de_type == 'KAZE':
        return cv2.KAZE_create(**de_params)
    else:
        raise ValueError('Unsupported descriptor extractor type \'%s\'.' % de_type)

def get_kp_and_des(ims, descriptor_extractor):
    logging.debug('Processing image descriptors...')
    sw = Stopwatch(); sw.start()
    
    keypoints, descriptors = [], []
    for i, im in enumerate(ims):
        kp, des = get_single_kp_and_des(im, descriptor_extractor)
        keypoints.append(kp)
        descriptors.append(des)
        
    sw.stop()
    logging.debug('Total number of keypoint+descriptor pairs: %d' % len(descriptors))
    logging.debug('Done processing image keypoints+descriptors. Took %s.' % sw.format_str())
    return keypoints, np.array(descriptors)

def get_single_kp_and_des(im, descriptor_extractor):
    kp, des = descriptor_extractor.detectAndCompute(im, None)
    if not kp:
        # kp and des are None => image is essentially blank
        kp, des = kp_and_des_for_blank_image(im, descriptor_extractor)
    kp = convert_cv2_kps_to_custom_kps(kp)
    return kp, des

def convert_cv2_kps_to_custom_kps(kps):
    return [KeyPoint_custom(kp) for kp in kps]

def kp_and_des_for_blank_image(im, descriptor_extractor):
    h, w = im.shape[:2]
    x, y = (int(w/2), int(h/2))
    size = 1
    kp = KeyPoint(x, y, size, -1, 0, 0, -1)
    des = np.zeros((descriptor_extractor.descriptorSize()), dtype=np.uint8)
    return [kp], [des]
    
def get_histograms(ims, BOVW, descriptor_extractor, spatial_pyramid_levels, n_bins_per_channel=4):

    features_string = 'BOVW+colors'
    logging.debug('Making %d %s histograms...' % (len(ims), features_string))
    sw = Stopwatch(); sw.start()

    histograms = []
    for im in ims:

        full_histogram, _, _ = \
            extract_features(im,
                             BOVW,
                             descriptor_extractor,
                             spatial_pyramid_levels,
                             n_bins_per_channel=n_bins_per_channel)

        histograms.append(full_histogram)
    all_histograms = np.vstack(histograms)

    sw.stop()
    logging.debug('Done making histograms. Took %s.' % sw.format_str())
    return all_histograms

def get_params_string(params):
    return ', '.join('%s=%s' % (k, v) for k,v in params.items()) if params else ''
    
def train(classifier, X, y):
    logging.debug('Fitting model...')
    sw = Stopwatch(); sw.start()
    
    classifier.fit(X, y)
    
    sw.stop()
    logging.debug('Done fitting model. Took %s.' % sw.format_str())
    

def get_labels_from_fps(im_fps):
    text_labels = np.array([im_fp.split('/')[-1].split('.')[0] for im_fp in im_fps]) # .../FD.6.png -> FD
    return np.array(text_labels)
  

def compute_error(im_labels, predictions):
    num_incorrect = sum(1 if l != p else 0 for (l, p) in zip(im_labels, predictions))
    num_total = len(im_labels)
    return num_incorrect / num_total

def get_score(Y, Y_hat):
    if len(Y) != len(Y_hat):
        raise ValueError
    return sum([1 if y == y_hat else 0 for y, y_hat in zip(Y, Y_hat)]) / len(Y)

def compute_average(l):
    return sum(l) / len(l)

def save_model(model, model_output_fp):
    model_output_dir = get_directory(model_output_fp)
    os.makedirs(model_output_dir, exist_ok=True)
    model.save(model_output_fp)

class KeyPoint_custom:
    def __init__(self, cv2_kp):
        self.pt = cv2_kp.pt
        self.x = cv2_kp.pt[0]
        self.y = cv2_kp.pt[1]
        self.size = cv2_kp.size
        self.angle = cv2_kp.angle
        self.class_id = cv2_kp.class_id
        self.response = cv2_kp.response
        self.octave = cv2_kp.octave

def convert_custom_kps_to_cv2_kps(kps):
    return [cv2.KeyPoint(kp.x, kp.y, kp.size, kp.angle, kp.response, kp.octave, kp.class_id) for kp in kps]
