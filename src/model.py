from cv2 import imread
from clustering import get_optimal_clustering
import sklearn; from sklearn.svm import SVC
from features import extract_features
import pickle
from f_hat import build_summed_area_table, f_hat
from ess import ESS


class Model(object):
    
    def __init__(self, descriptor_extractor):
        self.descriptor_extractor = descriptor_extractor
        
        
    def BOVW_create(self, im_fps, descriptor_extractor):

        try:
            ims = self.import_images(im_fps)
            kps, des_lists = zip(*self.get_descriptors(ims, descriptor_extractor))
            all_descriptors = [des for des_list in des_lists for des in des_list]
            bovw = get_optimal_clustering(all_descriptors)

            self.BOVW = bovw
            return True
        
        except Exception as e:
            print(e)
            return False
    
    
    def SVM_train(self, im_fps):
        
        try:
            ims = self.import_images(im_fps)
            labels = [im_fp.split('/')[-2] for im_fp in im_fps] # label = image folder

            histograms = self.get_histograms(ims, self.BOVW, descriptor_extractor)

            svm = SVC(kernel='rbf', C=0.1) # TODO: find optimal C;   TODO: support different kernels
            svm.fit(histograms, labels)

            self.SVM = svm
            return True
        
        except Exception as e:
            print(e)
            return False
    
    def get_histograms(self, ims):
        for im in ims:
            histogram, _ = extract_features(im, self.BOVW, self.descriptor_extractor)
            yield histogram
    
    def localize_w_ESS(self, im):
        
        _, cluster_matrix = extract_features(im, self.BOVW, self.descriptor_extractor)
        SAT = build_summed_area_table(cluster_matrix, self.SVM)
        
        bounding_box = ESS(im, f_hat, SAT)
        return bounding_box
    

    @staticmethod
    def import_images(im_fps):
        for im_fp in im_fps:
            yield imread(im_fp, 0) # TODO: add support for color

    @staticmethod
    def get_descriptors(ims, descriptor_extractor):
        for im in ims:
            yield descriptor_extractor.detectAndCompute(im, None)
        
        
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

