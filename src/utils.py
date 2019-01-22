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
        ims.append(imread(im_fp, 0))
    sw.stop()
    
    print('Import took %s.' % sw.format_str())
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
    print('Done processing image descriptors. Took %s.' % sw.format_str())
    return descriptors

    
def get_histograms(ims, BOVW, descriptor_extractor):
    print('Making BOVW histograms...')

    sw = Stopwatch(); sw.start()
    histograms = []
    for im in ims:
        histogram, _ = extract_features(im, BOVW, descriptor_extractor)
        histograms.append(histogram)
    vstacked = np.vstack(histograms)
    sw.stop()
    
    print('Done making histograms. Took %s.' % sw.format_str())
    return vstacked
    
    
def train(classifier, X, y):
    print('Fitting model...')
    sw = Stopwatch(); sw.start()
    
    classifier.fit(X, y)
    
    sw.stop()
    print('Done fitting model. Took %s.' % sw.format_str())
    

def get_labels_from_fps(im_fps):
    text_labels = np.array([im_fp.split('/')[-1].split('.')[0] for im_fp in im_fps]) # .../FD.6.png -> FD
    return text_labels
    
    
def get_score(Y, Y_hat):
    if len(Y) != len(Y_hat):
        raise ValueError
    return sum([1 if y == y_hat else 0 for y, y_hat in zip(Y, Y_hat)]) / len(Y)
