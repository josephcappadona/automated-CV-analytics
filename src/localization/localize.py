from ess import ESS
from f_hat import build_summed_area_table, f_hat
import sys, os
import nump as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features import extract_features


def localize_w_ESS(model, im, mask=None):

    (bovw_histogram, cluster_matrix), (color_histogram, _) = extract_features(im, model.BOVW, model.descriptor_extractor)
    histogram = np.vstack([bovw_histogram, color_histogram])

    prediction_text = model.SVM.predict([histogram])[0]
    prediction_index = model.SVM.label_binarizer_.transform([prediction_text]).indices[0]

    SAT = build_summed_area_table(cluster_matrix, model.SVM.estimators_[prediction_index])

    bounding_box = ESS(im, f_hat, SAT)
    return bounding_box
