import logging
import pickle
import numpy as np
import clustering
import cv2
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.kernel_approximation import RBFSampler, AdditiveChi2Sampler
from features import extract_features
import utils
from utils import Stopwatch


class Model(object):
    
    def __init__(self, model_type='LOGREG', model_params={}, BOVW_size=256, consider_colors=True, descriptor_extractor_type='ORB', descriptor_extractor_params={}, preprocess_params={}, data_transform=None, data_transform_params={}, feature_selection=None, feature_selection_params={}, approximation_kernel=None, approximation_kernel_params={}, **extras):

        self.model = None
        self.model_type = model_type
        self.model_params = model_params

        self.BOVW = None
        self.bovw_size = BOVW_size
        self.consider_colors = consider_colors

        self.descriptor_extractor = utils.create_descriptor_extractor(
                                        descriptor_extractor_type,
                                        descriptor_extractor_params
                                    )
        self.descriptor_extractor_type = descriptor_extractor_type
        self.descriptor_extractor_params = descriptor_extractor_params

        self.preprocess_params = preprocess_params

        self.transformer = None
        self.data_transform = data_transform
        self.data_transform_params = data_transform_params

        self.selector = None
        self.feature_selection = feature_selection
        self.feature_selection_params = feature_selection_params

        self.approx_kernel_map = None
        self.approximation_kernel = approximation_kernel
        self.approximation_kernel_params = approximation_kernel_params


    def BOVW_create(self, ims, show=False):
        logging.debug('Total images to process for BOVW: %d' % len(ims))
       
        ims = utils.preprocess_images(ims, **self.preprocess_params)
        descriptors = utils.get_descriptors(ims, self.descriptor_extractor)
        
        if type(self.bovw_size) == int:
            bovw = clustering.get_clustering(descriptors, self.bovw_size)
        elif type(self.bovw_size) == list:
            bovw = clustering.get_optimal_clustering(descriptors, cluster_sizes=self.bovw_size, show=show)
        else: # use default
            bovw = clustering.get_optimal_clustering(descriptors, show=show)

        self.BOVW = bovw
        logging.debug('BOVW (k=%d) created.' % self.BOVW.n_clusters)
        
    
    def train(self, train_ims, train_im_labels):

        train_ims = utils.preprocess_images(train_ims, **self.preprocess_params)
        train_im_histograms = \
            utils.get_histograms(train_ims,
                                 self.BOVW,
                                 self.descriptor_extractor,
                                 self.consider_colors)

        self.generate_data_transformers()
        train_im_histograms = self.fit_data_transformers(train_im_histograms)
        self.train_model(train_im_histograms, train_im_labels)
        
        return True
        
    # Model Selection (SVM, KNN)
    def train_model(self, train_im_histograms, train_im_labels):
        logging.debug('Training %s model with parameters %s...' % (self.model_type, utils.get_params_string(self.model_params)))

        if self.model_type == 'SVM':
            model = OneVsRestClassifier(SVC(**self.model_params))
        elif self.model_type == 'KNN':
            model = KNeighborsClassifier(**self.model_params)
        elif self.model_type == 'LOGREG':
            model = LogisticRegression(**self.model_params)
        else:
            raise ValueError('Unsupported decision model type \'%s\'.' % self.model_type)

        utils.train(model, train_im_histograms, train_im_labels)

        self.histograms = train_im_histograms
        self.labels = train_im_labels
        self.model = model


    def predict(self, test_ims):

        test_ims = utils.preprocess_images(test_ims, **self.preprocess_params)
        test_histograms = \
            utils.get_histograms(test_ims,
                                 self.BOVW,
                                 self.descriptor_extractor,
                                 consider_colors=self.consider_colors)
        test_histograms = self.transform_histograms(test_histograms)

        return self.model.predict(test_histograms)


    def generate_data_transformers(self):
        # Data Transformation (Scaling, Normalization)
        if self.data_transform:
            if self.data_transform == 'EXP':
                transformer = ''
                transformer.name = ''

            elif data_transform == 'NORM':
                pass

            transformer.params = utils.get_params_string(self.data_transform_params)
            self.transformer = transformer

        # Feature Selection (Var, Chi^2)
        if self.feature_selection:
            if self.feature_selection == 'VAR':
                selector = VarianceThreshold(**self.feature_selection_params)
                selector.name = 'VarianceThreshold'

            elif self.feature_selection == 'CHI2':
                pass

            selector.params = utils.get_params_string(self.feature_selection_params)
            self.selector = selector

        # Kernel Approximation (RBF, Chi^2)
        if self.approximation_kernel:
            if self.approximation_kernel == 'RBF':
                approx_kernel_map = RBFSampler(**self.kernel_approximation_params) 
                approx_kernel_map.name = 'RBFSampler'

            elif self.approximation_kernel == 'CHI2':
                approx_kernel_map = AdditiveChi2Sampler(**self.kernel_approximation_params) 
                approx_kernel_map.name = 'AdditiveChi2Sampler'

            approx_kernel_map.params = utils.get_params_string(self.kernelapproximation_params)
            self.approx_kernel_map = approx_kernel_map
      
    def fit_data_transformers(self, train_im_histograms):
        if self.transformer:
            train_im_histograms = self.transformer.transform(train_im_histograms)
        if self.selector:
            dim_before = len(train_im_histograms[0])
            train_im_histograms = self.selector.fit_transform(train_im_histograms)
            dim_after = len(train_im_histograms[0])
            logging.debug('Fit feature selector, reduced # of features from %d to %d' % (dim_before, dim_after))
        if self.approx_kernel_map:
            train_im_histograms = self.approx_kernel_map.fit_transform(train_im_histograms)
        return train_im_histograms

    def transform_histograms(self, im_histograms):
        if self.transformer:
            logging.debug('Transforming histograms with %s(%s)...' % (self.transformer.name, self.transformer.params)) 
            im_histograms = self.transformer.transform(im_histograms)

        if self.selector:
            logging.debug('Selecting features with %s(%s)...' % (self.selector.name, self.selector.params)) 
            im_histograms = self.selector.transform(im_histograms)

        if self.approx_kernel_map:
            logging.debug('Transforming histograms with %s(%s)...' % (self.approx_kernel_map.name, self.approx_kernel_map.params))
            im_histograms = self.approx_kernel_map.transform(im_histograms)

        return im_histograms
 
          
    def save(self, fp):
        d_e = self.descriptor_extractor
        self.descriptor_extractor = None # to prevent pickle error
        
        pickle.dump(self, open(fp, 'w+b'))
        self.descriptor_extractor = d_e # reset, in case we want to continue using model

    @staticmethod
    def load(model_fp):
        model = pickle.load(open(model_fp, 'rb'))
        model.descriptor_extractor = \
            utils.create_descriptor_extractor(
                model.descriptor_extractor_type,
                model.descriptor_extractor_params
            )
        return model

