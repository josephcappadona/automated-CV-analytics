from ess import ESS
from f_hat import build_summed_area_table, f_hat

def localize_w_ESS(model, im, mask=None):

	(bovw_histogram, cluster_matrix), (color_histogram, color_matrix) = \
		extract_features(im, self.BOVW, self.descriptor_extractor, mask=mask)

	prediction_text = self.SVM.predict([histogram])[0]
	prediction_index = self.SVM.label_binarizer_.transform([prediction_text]).indices[0]

	SAT = build_summed_area_table(cluster_matrix, self.SVM.estimators_[prediction_index])

	bounding_box = ESS(im, f_hat, SAT)
	return bounding_box

