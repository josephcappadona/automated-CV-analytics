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
            train_im_histograms = self.get_histograms(train_ims)
            
            # TODO: find optimal C;   TODO: support different kernels
            svm = OneVsRestClassifier(SVC(kernel='linear', C=0.1))
            svm.fit(train_im_histograms, train_im_labels)
            
            
            self.svm_histograms = train_im_histograms
            self.svm_labels = train_im_labels
            self.SVM = svm
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
        histograms = []
        for im in ims:
            histogram, _ = extract_features(im, self.BOVW, self.descriptor_extractor)
            histograms.append(histogram)
        return np.vstack(histograms)

    
    def localize_w_ESS(self, im):
        
        _, cluster_matrix = extract_features(im, self.BOVW, self.descriptor_extractor)
        SAT = build_summed_area_table(cluster_matrix, self.SVM)
        
        bounding_box = ESS(im, f_hat, SAT)
        return bounding_box
    

    @staticmethod
    def get_descriptors(ims, descriptor_extractor):
        print('Processing image descriptors...')
        count = 0
        for im in ims:
            if count % 10 == 0:
                print(count)
            yield descriptor_extractor.detectAndCompute(im, None)
            count += 1
        print('Done processing image descriptors.')
        
        
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

