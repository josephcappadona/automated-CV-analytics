import cv2
#from cv2.xfeatures2d import SIFT_create, SURF_create
from cv2 import ORB_create

# https://docs.opencv.org/3.4.2/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html
#sift = SIFT_create(nfeatures=5000, contrastThreshold=0.01, edgeThreshold=20, sigma=1.6)

# https://docs.opencv.org/3.4.2/d5/df7/classcv_1_1xfeatures2d_1_1SURF.html
#surf = SURF_create()

# https://docs.opencv.org/3.4.2/db/d95/classcv_1_1ORB.html
def orb_create():
    return ORB_create(5000, edgeThreshold=10, patchSize=10, scaleFactor=1.2, nlevels=8)
