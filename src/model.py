import pickle
import numpy as np
import clustering
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.kernel_approximation import RBFSampler, AdditiveChi2Sampler
from sklearn.multiclass import OneVsRestClassifier
from features import extract_features
from f_hat import build_summed_area_table, f_hat
from ess import ESS
from time import time
from utils import get_descriptors, get_histograms, Stopwatch, train


class Model(object):
    
    def __init__(self, descriptor_extractor_create):
        self.descriptor_extractor_create = descriptor_extractor_create
        self.descriptor_extractor = descriptor_extractor_create()
        self.BOVW = None
        self.SVM = None
        self.approx_kernel_map = None
        
        
    def BOVW_create(self, ims, k=None, show=False):
        print('Total images to process (in training set): %d' % len(ims))
        
        descriptors = get_descriptors(ims, self.descriptor_extractor)
        
        if k:
            bovw = clustering.get_optimal_clustering(descriptors, cluster_sizes=k, show=show)
        else:
            bovw = clustering.get_optimal_clustering(descriptors, show=show)

        self.BOVW = bovw
        return True
        
    
    def train(self, model_type, train_ims, train_im_labels, consider_descriptors=True, consider_colors=True, kernel_approx=None):

        self.consider_descriptors = consider_descriptors
        self.consider_colors = consider_colors

        train_im_histograms = get_histograms(train_ims,
                                             self.BOVW,
                                             self.descriptor_extractor,
                                             consider_descriptors=consider_descriptors,
                                             consider_colors=consider_colors)
        
        # Kernel Approximation (RBF, Chi^2)
        if kernel_approx:
            if kernel_approx == 'rbf':
                gamma = 2
                n_components = len(train_im_histograms[0])*20
                approx_kernel_map = RBFSampler(gamma=gamma, n_components=n_components)
                
                kernel_name = 'RBFSampler'
                kernel_params = 'gamma=%g, n_components=%d' % (gamma, n_components)
                
            elif kernel_approx == 'chi2':
                sample_steps = 1
                sample_interval = None
                approx_kernel_map = AdditiveChi2Sampler(sample_steps=sample_steps, sample_interval=sample_interval)
                
                kernel_name = 'AdditiveChi2Sampler'
                kernel_params = 'sample_steps=%d, sample_interval=%s' % (sample_steps, sample_interval)
                
            print('Transforming histograms with %s(%s)...' % (kernel_name, kernel_params))
            train_im_histograms = approx_kernel_map.fit_transform(train_im_histograms)
            self.approx_kernel_map = approx_kernel_map
        
        # Model Selection (SVM, KNN)
        if model_type.lower() == 'svm':
            svm = OneVsRestClassifier(SVC(kernel='linear', C=100)) # C=100 b/c Chapelle et al
            train(svm, train_im_histograms, train_im_labels)

            self.svm_histograms = train_im_histograms
            self.svm_labels = train_im_labels
            self.SVM = svm
        elif model_type.lower() == 'knn':
            knn = KNeighborsClassifier()
            train(knn, train_im_histograms, train_im_labels)
            
            self.knn_histograms = train_im_histograms
            self.knn_labels = train_im_labels
            self.KNN = knn
        return True
        
        
    def predict(self, model_type, test_ims, masks=None):
        
        test_histograms = get_histograms(test_ims,
                                         self.BOVW,
                                         self.descriptor_extractor,
                                         masks=masks,
                                         consider_descriptors=self.consider_descriptors,
                                         consider_colors=self.consider_colors)
        if self.approx_kernel_map:
            test_histograms = self.approx_kernel_map.transform(test_histograms)
        
        if model_type.lower() == 'svm':
            return self.SVM.predict(test_histograms)
        elif model_type.lower() == 'knn':
            return self.KNN.predict(test_histograms)
            

    def localize_w_ESS(self, im, mask=None):
        
        (bovw_histogram, cluster_matrix), (color_histogram, color_matrix) = \
            extract_features(im,
                             self.BOVW,
                             self.descriptor_extractor,
                             mask=mask)
        
        y_hat_text = self.SVM.predict([histogram])[0]
        y_hat_index = self.SVM.label_binarizer_.transform([y_hat_text]).indices[0]
        
        SAT = build_summed_area_table(cluster_matrix, self.SVM.estimators_[y_hat_index])
        
        bounding_box = ESS(im, f_hat, SAT)
        return bounding_box
        
        
    def save_model(self, fp):

        d_e = self.descriptor_extractor
        self.descriptor_extractor = None # to prevent pickle error
        
        pickle.dump(self, open(fp, 'w+b'))
        self.descriptor_extractor = d_e # reset, in case we want to continue using model

    @staticmethod
    def load_model(model_fp):
        model = pickle.load(open(model_fp, 'rb'))
        model.descriptor_extractor = model.descriptor_extractor_create()
        return model

