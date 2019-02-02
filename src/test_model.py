from model import Model
from descriptor_extractors import orb_create
from random import shuffle
from sys import argv
from glob import iglob
from pickle import load
from sklearn.model_selection import train_test_split
from cv2 import imread
import numpy as np
from time import localtime, strftime
import warnings; warnings.filterwarnings('ignore')
from utils import get_directory, import_images, get_labels_from_fps, get_score
from args import parse_args


args = parse_args(argv)

usage = "\nUSAGE:  python test_model.py -d DATA_DIR [-o MODEL_OUTPUT_FP | -m MODEL_INPUT_FP] [--consider_descriptors 0/1] [--consider_colors 0/1] [--test_size 0.??]\n\nIf -m argument is specified, model is loaded from file and tested on data in DATA_DIR. If -m is not specified, a new model is created from from the data in DATA_DIR.\n\nDefault values if argument not specified:\n-o \"output/model %Y-%m-%d %H:%M:%S.pkl\"\n--consider_descriptors 1\n--consider_colors 1\n--test_size 0.8\n"
if not args['d']: # if no DATA_DIR specified
    print('No DATA_DIR specified.\n')
    print(usage)
    exit()


im_dir = args['d']
model_output_fp = args['o'] if args['o'] \
                      else ('output/model %s.pkl' % \
                            strftime('%Y-%m-%d %H:%M:%S', localtime()))
model_input_fp = args['m']

consider_descriptors = bool(int(args['consider_descriptors'])) \
                           if args['consider_descriptors'] else True
consider_colors = bool(int(args['consider_colors'])) \
                      if args['consider_colors'] else True
test_size = float(args['test_size']) if args['test_size'] else 0.80


im_fps = iglob(im_dir + '/*_snippets/*.png')

if not model_input_fp:
    print('Creating new model...')
     
    print('Loading data...')
    X = import_images(im_fps)
    y = get_labels_from_fps(im_fps)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    print('Building model...')
    m = Model(orb_create)
    m.BOVW_create(X_train, k=[32], show=True)
    m.SVM_train(X_train, y_train)

    print('Saving model...')
    model_output_dir = get_directory(model_output_fp)
    makedirs(model_output_dir, exist_ok=True)
    m.save(model_output_fp)
    
elif model_input_fp:
    print('Loading model from file...')
    m = load(open(model_input_fp, 'rb'))

    print('Loading data...')
    X_test = import_images(im_fps)
    y_test = get_labels_from_fps(im_fps)
    
if test_size:
    print('Predicting...')
    y_hat = m.SVM_predict(X_test)
    print('Accuracy (test set): %.3f' % get_score(y_test, y_hat))

import code; code.interact(local=locals())

