import time
from cv2 import imread
import numpy as np
from features import extract_features


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
        m = int(self.duration / 60); m_str = ('%dm' % m) if m > 0 else ''
        s = int(self.duration % 60); s_str = '%ds' % s
        return m_str + s_str


# TODO: add support for color
def import_images(im_fps): 
    print('Importing images...\nTotal images: %d' % len(im_fps))

    sw = Stopwatch(); sw.start()
    ims = []
    n = int(len(im_fps) / 10)
    for i, im_fp in enumerate(im_fps):
        if i % n == 0:
            print(i)
        ims.append(imread(im_fp))
    sw.stop()
    
    print('Import took %s.\n\n' % sw.format_str())
    return ims


def get_descriptors(ims, descriptor_extractor):
    print('Processing image descriptors...')
    sw = Stopwatch(); sw.start()
    
    descriptors = []
    n = int(len(ims) / 10)
    for i, im in enumerate(ims):
        if i % n == 0:
            print(i)
        kp, des = descriptor_extractor.detectAndCompute(im, None)
        descriptors.extend(des)
        
    sw.stop()
    print('Total number of descriptors: %d' % len(descriptors))
    print('Done processing image descriptors. Took %s.\n\n' % sw.format_str())
    return descriptors

    
def get_histograms(ims, BOVW, descriptor_extractor, n_bins_per_color=4, masks=None, consider_descriptors=True, consider_colors=True):
    print('Making BOVW histograms...')
    sw = Stopwatch(); sw.start()

    histograms = []
    for i, im in enumerate(ims):
        mask = masks[i] if masks else None

        (bovw_histogram, _), (color_histogram, _) = \
            extract_features(im,
                             BOVW,
                             descriptor_extractor,
                             n_bins_per_color=n_bins_per_color,
                             mask=mask)

        if consider_descriptors and consider_colors:
            complex_histogram = np.hstack((bovw_histogram, color_histogram))
            histograms.append(complex_histogram)
        elif consider_descriptors:
            histograms.append(bovw_histogram)
        elif consider_colors:
            histograms.append(color_histogram)
        else:
            raise ValueError("Histogram is empty (neither descriptors nor colors are being considered).")
    vstacked = np.vstack(histograms)

    sw.stop()
    print('Done making histograms. Took %s.\n\n' % sw.format_str())
    return vstacked
    
    
def train(classifier, X, y):
    print('Fitting model...')
    sw = Stopwatch(); sw.start()
    
    classifier.fit(X, y)
    
    sw.stop()
    print('Done fitting model. Took %s.\n\n' % sw.format_str())
    

def get_labels_from_fps(im_fps):
    text_labels = np.array([im_fp.split('/')[-1].split('.')[0] for im_fp in im_fps]) # .../FD.6.png -> FD
    return text_labels
    
    
def get_score(Y, Y_hat):
    if len(Y) != len(Y_hat):
        raise ValueError
    return sum([1 if y == y_hat else 0 for y, y_hat in zip(Y, Y_hat)]) / len(Y)
