from model import Model
from descriptor_extractors import orb
from random import shuffle
from sys import argv
from glob import iglob
from pickle import load
from sklearn.model_selection import train_test_split
from cv2 import imread
import numpy as np

def import_images(im_fps):
    # TODO: add support for color
    return [imread(im_fp, 0) for im_fp in im_fps]

def get_labels_from_fps(im_fps):
    text_labels = np.array([im_fp.split('/')[-1].split('.')[0] for im_fp in im_fps]) # .../FD.6.png -> FD
    return text_labels
    
def get_score(Y, Y_hat):
    if len(Y) != len(Y_hat):
        raise ValueError
    return sum([1 if y == y_hat else 0 for y, y_hat in zip(Y, Y_hat)]) / len(y)


# parse_args -> -o model_output_fp, -d data_dir, -m model_input_fp, 


if len(argv) == 2:
    print('Creating new model...')
    m = Model(orb)
    
    im_dir = argv[1]
    im_fps = list(iglob(im_dir + '/*_snippets/*.png'))
    
    print('Loading data...')
    X = import_images(im_fps)
    y = get_labels_from_fps(im_fps)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    m.BOVW_create(X_train, k=[64], show=False)
    m.SVM_train(X_train, y_train)
    
elif len(argv) == 3:
    print('Loading model from file...')
    try:
        model_fp = argv[2]
        m = load(open(model_fp, 'rb'))
    except Exception as e:
        print('ERROR: Could not load model (%s) - %s' % (model_fp, e))
        exit()
        
    im_dir = argv[1]
    im_fps = list(iglob(im_dir + '/*_snippets/*.png'))
    
    print('Loading data...')
    X_test = import_images(im_fps)
    y_test = get_labels_from_fps(im_fps)
    
else:
    print('USAGE:  python test_model.py DATA_PATH [MODEL_PATH]\n')
    exit()


print('Predicting...')
y_hat = m.SVM_predict(X_test)
print(get_score(y_test, y_hat))

import code; code.interact(local=locals())

