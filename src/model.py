import pickle
import numpy as np
import clustering
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from features import extract_features
from f_hat import build_summed_area_table, f_hat
from ess import ESS
from time import time
from utils import get_descriptors, get_histograms, Stopwatch, train


class Model(object):
    
    def __init__(self, descriptor_extractor_create):
        self.descriptor_extractor_creator = descriptor_extractor_create
        self.descriptor_extractor = descriptor_extractor_create()
        
        
    def BOVW_create(self, ims, k=None, show=True):
        print('Creating BOVW...')
        print('Total images to process: %d' % len(ims))
        
        descriptors = get_descriptors(ims, self.descriptor_extractor)
        
        if k:
            bovw = clustering.get_optimal_clustering(descriptors, cluster_sizes=k, show=show)
        else:
            bovw = clustering.get_optimal_clustering(descriptors, show=show)

        self.BOVW = bovw
        return True
        
    
    def SVM_train(self, train_ims, train_im_labels, consider_descriptors=True, consider_colors=True):
        print('Training SVM decision model...')

        train_im_histograms = get_histograms(train_ims,
                                             self.BOVW,
                                             self.descriptor_extractor,
                                             consider_descriptors=consider_descriptors,
                                             consider_colors=consider_colors)
        
        # TODO: find optimal C;   TODO: support different kernels
        svm = OneVsRestClassifier(SVC(kernel='linear', C=0.1))
        train(svm, train_im_histograms, train_im_labels)
        
        
        self.svm_histograms = train_im_histograms
        self.svm_labels = train_im_labels
        self.SVM = svm    
        return True
        
        
    def SVM_predict(self, test_ims, masks=None, consider_descriptors=True, consider_colors=True):
        
        test_histograms = get_histograms(test_ims,
                                         self.BOVW,
                                         self.descriptor_extractor,
                                         masks=masks,
                                         consider_descriptors=consider_descriptors,
                                         consider_colors=consider_colors)
        return self.SVM.predict(test_histograms)
            

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
        
        
    def save(self, fp):
        d_e = self.descriptor_extractor
        self.descriptor_extractor = None # to prevent pickle error
        try:
            pickle.dump(self, open(fp, 'w+b'))
            self.descriptor_extractor = d_e # reset, in case we want to continue using model
            return True
        except Exception as e:
            print(e)
            return False

    @staticmethod
    def import_model(model_fp):
        model = pickle.load(open(model_fp, 'rb'))
        model.descriptor_extractor = model.descriptor_extractor_create()
        return model

