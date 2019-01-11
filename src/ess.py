from queue import PriorityQueue
from math import floor
from f_hat import f_hat

def ESS(im, f_hat, SAT):
    
    # initialize data structures
    PQ = PriorityQueue()

    h, w = im.shape[:2]
    T = [0, h-1]
    B = [0, h-1]
    L = [0, w-1]
    R = [0, w-1]
    bounds = [T, B, L, R]
    

    # loop until global optimum is found
    while not is_single_rectangle(bounds):
        new_bounds = split_bounds(bounds)
        for new_bound in new_bounds:
            PQ.push(f_hat(new_bound, SAT), new_bound)
        bounds = PQ.get()
    # TODO: mask optimum region and try again

    # return bounding box
    T, B, L, R = bounds
    return (T[0], B[0], L[0], R[0])


def is_single_rectangle(bounds):
    T, B, L, R = bounds
    return T[0] == T[1] and \
           B[0] == B[1] and \
           L[0] == L[1] and \
           R[0] == R[1]

            
def split_bounds(bounds):
    T, B, L, R = bounds
    new_bounds = []

    # split top margin interval (if possible)
    if T[0] != T[1]:
        T_1, T_2 = split_interval(T)
        new_bounds.append([T_1, B, L, R])
        new_bounds.append([T_2, B, L, R])

    # split bottom margin interval
    if B[0] != B[1]:
        B_1, B_2 = split_interval(B)
        new_bounds.append([T, B_1, L, R])
        new_bounds.append([T, B_2, L, R])

    # split left margin interval
    if L[0] != L[1]:
        L_1, L_2 = split_interval(L)
        new_bounds.append([T, B, L_1, R])
        new_bounds.append([T, B, L_2, R])

    # split right margin interval
    if R[0] != R[1]:
        R_1, R_2 = split_interval(R)
        new_bounds.append([T, B, L, R_1])
        new_bounds.append([T, B, L, R_2])

        
def split_interval(I):
    I_1_lo = I[0]
    I_1_hi = int(floor((I[0] + I[1])/2))
    I_2_lo = I_1_hi + 1
    I_2_hi = I[1]

    I_1 = [I_1_lo, I_1_hi]
    I_2 = [I_2_lo, I_2_hi]
    return I_1, I_2
