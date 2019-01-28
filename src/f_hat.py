import numpy as np

def f_hat(bounds, SAT):
    """Computes f_hat(y), an upper bound for f(y), where y is an interval bound for a set of rectangles

        In particular,
                f(y) = \sum_{d_i \in D}{w_{c_i}} for D_i = the ith descriptor, corresponding to the ith keypoint, belonging to cluster_id c_i;
        and,
            as shown in Lampert et al (Eq 5), f_hat(Y) = f^+(y_\cup) + f^-(y_\cap) >= f(y) for all y \in Y, where f^+ contains only the positive summands w_{c_i} and f^- contains only the negatives summands w_{c_i}, and y_\cup corresponds to the union of all rectangles in y and y_\cap corresponds to the intersection of all rectangles in y
            
            
    Args:
        bounds (list): a set of rectangles, representing candidate bounding boxes, in the form T x B x L x R, where T, B, L, and R are interval bounds for the top, bottom, left, and right margins of the set of bounding boxes
            structure is [[top_min, top_max], [bottom_min, bottom_max], [left_min, left_max], [right_min, right_max]] 
        SAT (n x m array):

    Returns:
        f_hat_(int): an upper bound for the quality function, f, over the input bounds


    Running time:
        Since we can find f^+(y_\cup) and f^-(y_\cap) in O(1) at runtime if we precompute integral images of the f^+ and f^-, implemented as summed area tables, we can compute f_hat in O(1)
    """
    
    union_box, intersection_box = get_union_and_intersection_bounding_boxes(bounds)
    f_pos_ = f_sum(union_box, SAT[:,:,1])
    f_neg_ = f_sum(intersection_box, SAT[:,:,0])
    
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

