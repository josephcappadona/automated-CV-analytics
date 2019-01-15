from cv2 import imread
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


class Model(object):
    
    def __init__(self, descriptor_extractor):
        self.descriptor_extractor = descriptor_extractor
        
        
    def BOVW_create(self, ims, k=None, show=True):

        try:
            print('Creating BOVW...')
            print('Total images to process: %d' % len(ims))
            
            kps, des_lists = zip(*self.get_descriptors(ims, self.descriptor_extractor))
            all_descriptors = [des for des_list in des_lists for des in des_list]
            print('Total number of descriptors: %d' % len(all_descriptors))
            if not k:
                bovw, k = clustering.get_optimal_clustering(all_descriptors, show=show)
            else:
                bovw, k = clustering.get_optimal_clustering(all_descriptors, cluster_sizes=k, show=show)

            self.BOVW = bovw
            return True
        
        except Exception as e:
            print('\nERROR: %s\n' % e)
            return False
    
    
    def SVM_train(self, train_ims, train_im_labels):
        
        try:
            print('Training SVM decision model...')
            start = time()

            train_im_histograms = self.get_histograms(train_ims)
            
            # TODO: find optimal C;   TODO: support different kernels
            print('Fitting model...')
            start_fit = time()
            svm = OneVsRestClassifier(SVC(kernel='linear', C=0.1))
            svm.fit(train_im_histograms, train_im_labels)
            end_fit = time()
            s_fit_total = int(end_fit - start_fit)
            m_fit = int(s_fit_total / 60)
            s_fit = int(s_fit_total % 60)
            print('Done fitting model. Took %dm%ds.' % (m_fit, s_fit))
            
            
            self.svm_histograms = train_im_histograms
            self.svm_labels = train_im_labels
            self.SVM = svm

            end = time()
            s_total = int(end - start)
            m = int(s_total / 60)
            s = s_total % 60
            print('Done training SVM model. Took %dm%ds.' % (m, s))
            return True
        
        except Exception as e:
            print(e)
            return False
        
    def SVM_predict(self, test_ims):
        
        try:
            test_histograms = self.get_histograms(test_ims)
            return self.SVM.predict(test_histograms)
        
        except Exception as e:
            print('ERROR: %s' % e)
            #return None
            raise(e)
            
    
    def get_histograms(self, ims):
        print('Making image BOVW histograms...')

        start = time()
        histograms = []
        for im in ims:
            histogram, _ = extract_features(im, self.BOVW, self.descriptor_extractor)
            histograms.append(histogram)
        vstacked = np.vstack(histograms)
        
        end = time()
        s_total = int(end - start)
        m = int(s_total / 60)
        s = s_total % 60
        print('Done making histograms. Took %dm%ds.' % (m, s))
        return vstacked

    
    def localize_w_ESS(self, im):
        
        _, cluster_matrix = extract_features(im, self.BOVW, self.descriptor_extractor)
        SAT = build_summed_area_table(cluster_matrix, self.SVM)
        
        bounding_box = ESS(im, f_hat, SAT)
        return bounding_box
    

    @staticmethod
    def get_descriptors(ims, descriptor_extractor):
        print('Processing image descriptors...')

        start = time()
        n = int(len(ims) / 10)
        for i, im in enumerate(ims):
            if i % n == 0:
                print(i)
            yield descriptor_extractor.detectAndCompute(im, None)
        end = time()
        s_total = int(end - start)
        m = int(s_total / 60)
        s = s_total % 60
        print('Done processing image descriptors. Took %dm%ds.' % (m, s))
        
        
    def save_model(self, fp):
        try:
            pickle.dump(self, open(fp, 'w+b'))
            return True
        except Exception as e:
            print(e)
            return False

    @staticmethod
    def import_model(model_fp):
        return pickle.load(model_fp)

