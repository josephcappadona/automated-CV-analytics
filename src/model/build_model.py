import argparse
import sys
import yaml
import logging
import utils
import time
import glob2
import os
import numpy as np
from model import Model
import hyperparameter_tuning
import pickle

# define command line args
parser = argparse.ArgumentParser(prog=('python %s' % sys.argv[0]), description='Build an image classification model.')
optional_args = parser._action_groups.pop()
required_args = parser.add_argument_group('required arguments')
parser._action_groups.append(optional_args)

optional_args.add_argument('-v', help='Verbose output (debug)', action='store_true')
optional_args.add_argument('-e', type=str, help='The output location of cross validation errors for each config, for analysis after training is complete', default='', metavar='ERROR_INFO_FP')
optional_args.add_argument('-store-all', help='Store all models (default: store only the best models)', action='store_true')

required_args.add_argument('-train', type=str, help='The location of the training snippets', required=True, metavar='TRAIN_DIR')
required_args.add_argument('-test', type=str, help='The location of the testing snippets', required=True, metavar='TEST_DIR')
required_args.add_argument('-c', type=str, help='The location of the config file with the model parameters', required=True, metavar='CONFIG')
required_args.add_argument('-m', type=str, help='The output location of the created model', required=True, metavar='MODEL_OUTPUT_FP')


# load command line args
args = parser.parse_args(sys.argv[1:] if len(sys.argv) > 1 else ['-h'])
train_dir = args.train
test_dir = args.train
config_fp = args.c
model_output_fp = args.m
verbose = args.v
store_all = args.store_all
error_info_fp = args.e

# set logger, start stopwatch
logging_level = logging.DEBUG if verbose else logging.INFO
logging.basicConfig(stream=sys.stdout, level=logging_level)
sw = utils.Stopwatch(); sw.start()

# load model config
logging.info('Loading model config...')
with open(config_fp, 'rt') as config_file:
    model_args = yaml.load(config_file)

# import training data
logging.info('Loading train data...')
train_im_fps = glob2.glob(os.path.join(train_dir, '**/snippets/**/*.png'))
train_ims = utils.import_images(train_im_fps)
train_im_labels = utils.get_labels_from_fps(train_im_fps)

# import testing data
logging.info('Loading test data...')
test_im_fps = glob2.glob(os.path.join(test_dir, '**/snippets/**/*.png'))
test_ims = utils.import_images(test_im_fps)
test_im_labels = utils.get_labels_from_fps(test_im_fps)

import pprint
def build_model(model_args, X, y):
    logging.info('Building new model...')
    model = Model(**model_args)

    logging.info('Building BOVW...')
    model.BOVW_create(X)

    logging.info('Training model...')
    model.train(X, y)

    return model

def cross_val_model(model_config, X, y):
    n_folds = model_config['n_folds']
    logging.info('Running k-fold cross-validation on model (k=%d)...' % n_folds)

    cross_val_errors = []
    for j, ((X_cv_train, y_cv_train), (X_cv_test, y_cv_test)) \
        in enumerate(hyperparameter_tuning.get_folds(X, y, n_splits=n_folds)):

        print(); logging.info('Fold #%d of %d' % (j+1, n_folds))

        model = build_model(model_config, X_cv_train, y_cv_train)
        val_error = 1 - test_model(model, X_cv_test, y_cv_test)
        cross_val_errors.append(val_error)

    avg_cross_val_error = utils.compute_average(cross_val_errors)
    print(); logging.info('Average cross validation error: %g\n' % avg_cross_val_error)
    return avg_cross_val_error

def test_model(model, X, y):
    logging.info('Testing model...')
    predictions = model.predict(X)
    test_acc = 1 - utils.compute_error(y, predictions)
    num_total = len(y)
    num_correct = int(round(test_acc * num_total))
    logging.info('Test accuracy: %g (%d/%d)' % (test_acc, num_correct, num_total))
    return test_acc

# 
model_configs = hyperparameter_tuning.get_model_configs(model_args)
all_models = hyperparameter_tuning.create_all_models(model_configs, build_model, cross_val_model, test_model, train_ims, train_im_labels, test_ims, test_im_labels)

best_models = hyperparameter_tuning.find_best_models(all_models)
best_test_acc = best_models[0][3]

#
print('\n\n'); logging.info('Saving models...')
with open(model_output_fp, 'w+b') as model_output_file:
    if store_all:
        pickle.dump(all_models, model_output_file)
        logging.info('All %d model(s) and train+test statistics saved to \'%s\'.' % (len(model_configs), model_output_fp))
    else:
        pickle.dump(best_models, model_output_file)
        logging.info('%d/%d model(s) (test_acc=%g) and train+test statistics saved to \'%s\'.' % (len(best_models), len(model_configs), best_test_acc, model_output_fp))

#
sw.stop()
print(); logging.info('Model creation took %s.' % sw.format_str())

'''
import visualize
bovw = model.BOVW
visualize.visualize_BOVW_PCA(bovw)
'''
