import numpy as np

def f_hat(bounds, SAT):
    
    union_box, intersection_box = get_union_and_intersection_bounding_boxes(bounds)
    f_pos_ = f_sum(union_box, SAT[:,:,1]) # SAT[:,:,1] corresponds to positive feature weights
    f_neg_ = f_sum(intersection_box, SAT[:,:,0]) # SAT[:,:,0] corresponds to negative feature weights
    
    f_hat_ = f_pos_ + f_neg_
    return f_hat_
    
def f_sum(box, SAT):
    
    top_left, bottom_left, top_right, bottom_right = box
    TL_x, TL_y = top_left
    TR_x, TR_y = top_right
    BL_x, BL_y = bottom_left
    BR_x, BR_y = bottom_right
    
    A = SAT[TL_x-1, TL_y-1] if TL_x != 0 and TL_y != 0 else 0
    B = SAT[TR_x, TR_y-1] if TR_y != 0 else 0
    C = SAT[BL_x-1, BL_y] if BL_x != 0 else 0
    D = SAT[BR_x, BR_y]
    
    area_sum = D - B - C + A
    return area_sum
    

def get_union_and_intersection_bounding_boxes(bounds):
    
    t_lo, t_hi = bounds[0]
    b_lo, b_hi = bounds[1]
    l_lo, l_hi = bounds[2]
    r_lo, r_hi = bounds[3]
    
    
    union_box = ((l_lo, t_lo), # top-left
                 (l_lo, b_hi), # bottom-left
                 (r_hi, t_lo), # top-right
                 (r_hi, b_hi)) # bottom-right
    
    intersection_box = ((l_hi, t_hi), # top-left
                        (l_hi, b_lo), # bottom-left
                        (r_lo, t_hi), # top-right
                        (r_lo, b_lo)) # bottom-right
    
    return (union_box, intersection_box)


# integral image over image's bovw clusters, weighted by SVM support vectors and coefficients
# used to quickly calculate the sums of histograms over areas
# TODO: native support for RBF kernel (not via kernel approx) https://stats.stackexchange.com/questions/86207/how-to-construct-the-feature-weight-vector-or-decision-boundary-from-a-linear
def build_summed_area_table(cluster_matrix, svm_model):
    
    h, w = cluster_matrix.shape[:2]
    SAT = np.zeros((h, w, 2), dtype=np.float)
    
    dual_coef = svm_model.dual_coef_
    sv = svm_model.support_vectors_
    
    for x in range(w):
        for y in range(h):
            
            c_i = cluster_matrix[y, x]
            w_i = dual_coef.dot(sv[:,c_i]) if c_i != -1 else 0
            # w_i = 0  =>  no descriptor at cluster_matrix[y,x]
            
            if w_i > 0:
                SAT[y, x, 0] = w_i
            elif w_i < 0:
                SAT[y, x, 1] = w_i
                
            if x != 0:
                SAT[y, x] += SAT[y, x-1]
            if y != 0:
                SAT[y, x] += SAT[y-1, x]
            if x !=0 and y != 0:
                SAT[y, x] -= SAT[y-1, x-1]
                
    return SAT

