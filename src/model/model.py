import logging
import pickle
import numpy as np
import clustering
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
    
    def __init__(self):
        
        self.model = None
        self.BOVW = None

        self.descriptor_extractor = None

        self.transformer = None
        self.selector = None
        self.approx_kernel_map = None


    def BOVW_create(self, ims, de_type, de_params, bovw_size, show=False):
        logging.debug('Total images to process for BOVW: %d' % len(ims))
       
        self.de_type, self.de_params = de_type, de_params
        self.descriptor_extractor = utils.create_descriptor_extractor(de_type, de_params)
        descriptors = utils.get_descriptors(ims, self.descriptor_extractor)
        
        if type(bovw_size) == int:
            bovw = clustering.get_clustering(descriptors, bovw_size) 
        elif type(bovw_size) == list:
            bovw = clustering.get_optimal_clustering(descriptors, cluster_sizes=bovw_size, show=show)
        else: # use default
            bovw = clustering.get_optimal_clustering(descriptors, show=show)

        self.BOVW = bovw
        logging.debug('BOVW (k=%d) created.' % bovw.n_clusters)
        
    
    def train(self,
              train_ims, train_im_labels,
              model_type='LogReg', model_params={},
              consider_descriptors=True, consider_colors=True,
              data_transform=None, data_transform_params={},
              feature_selection=None, feature_selection_params={},
              approximation_kernel=None, kernel_approx_params={},
              **extra_args):

        self.consider_descriptors = consider_descriptors
        self.consider_colors = consider_colors

        train_im_histograms = \
            utils.get_histograms(train_ims,
                                 self.BOVW,
                                 self.descriptor_extractor,
                                 consider_descriptors=consider_descriptors,
                                 consider_colors=consider_colors)

        self.generate_data_transformers(data_transform, data_transform_params,
                                        feature_selection, feature_selection_params,
                                        approximation_kernel, kernel_approx_params)
        train_im_histograms = self.fit_data_transformers(train_im_histograms)
        self.train_model(model_type, model_params,
                         train_im_histograms, train_im_labels)
        
        return True
        
    # Model Selection (SVM, KNN)
    def train_model(self, model_type, model_params, train_im_histograms, train_im_labels):
        logging.debug('Training %s model with parameters %s...' % (model_type, utils.get_params_string(model_params)))

        self.model_type = model_type
        if model_type == 'SVM':
            model = OneVsRestClassifier(SVC(**model_params))
        elif model_type == 'KNN':
            model = KNeighborsClassifier(**model_params)
        elif model_type == 'LOGREG':
            model = LogisticRegression(**model_params)
        else:
            raise ValueError('Unsupported decision model type \'%s\'.' % self.model_type)

        utils.train(model, train_im_histograms, train_im_labels)

        self.histograms = train_im_histograms
        self.labels = train_im_labels
        self.model = model


    def predict(self, test_ims, masks=None):
        
        test_histograms = \
            utils.get_histograms(test_ims,
                                 self.BOVW,
                                 self.descriptor_extractor,
                                 masks=masks,
                                 consider_descriptors=self.consider_descriptors,
                                 consider_colors=self.consider_colors)

        test_histograms = self.transform_histograms(test_histograms)

        return self.model.predict(test_histograms)


    def generate_data_transformers(self, data_transform, data_transform_params, feature_selection, feature_selection_params, approximation_kernel, kernel_approx_params):
        # Data Transformation (Scaling, Normalization)
        if data_transform:
            if data_transform == 'EXP':
                transformer = ''
                transformer.name = ''

            elif data_transform == 'NORM':
                pass

            transformer.params = utils.get_params_string(data_transform_params)
            self.transformer = transformer

        # Feature Selection (Var, Chi^2)
        if feature_selection:
            if feature_selection == 'VAR':
                selector = VarianceThreshold(**feature_selection_params)
                selector.name = 'VarianceThreshold'

            elif feature_selection == 'CHI2':
                pass

            selector.params = utils.get_params_string(feature_selection_params)
            self.selector = selector

        # Kernel Approximation (RBF, Chi^2)
        if approximation_kernel:
            if approximation_kernel == 'RBF':
                approx_kernel_map = RBFSampler(**approx_kernel_params) 
                approx_kernel_map.name = 'RBFSampler'

            elif approximation_kernel == 'CHI2':
                approx_kernel_map = AdditiveChi2Sampler(**approx_kernel_params) 
                approx_kernel_map.name = 'AdditiveChi2Sampler'

            approx_kernel_map.params = utils.get_params_string(approx_kernel_params)
            self.approx_kernel_map = approx_kernel_map
      
    def fit_data_transformers(self, train_im_histograms):
        if self.transformer:
            train_im_histograms = self.transformer.transform(train_im_histograms)
        if self.selector:
            before = len(train_im_histograms[0])
            train_im_histograms = self.selector.fit_transform(train_im_histograms)
            after = len(train_im_histograms[0])
            logging.debug('Fit feature selector, reduced # of features from %d to %d' % (before, after))
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
        model.descriptor_extractor = utils.create_descriptor_extractor(model.de_type, model.de_params)
        return model

