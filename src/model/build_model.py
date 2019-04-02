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
import random
import pprint
import pickle


parser = argparse.ArgumentParser(prog=('python %s' % sys.argv[0]), description='Build an image classification model.')
optional_args = parser._action_groups.pop()
required_args = parser.add_argument_group('required arguments')
required_args.add_argument('-d', type=str, help='The location of the training snippets', required=True, metavar='DATA_DIR')
required_args.add_argument('-c', type=str, help='The location of the config file with the model parameters', required=True, metavar='CONFIG')
required_args.add_argument('-m', type=str, help='The output location of the created model', required=True, metavar='MODEL_OUTPUT_FP')
optional_args.add_argument('-v', help='Verbose output (debug)', action='store_true')
optional_args.add_argument('-e', type=str, help='The output location of cross validation errors for each config, for analysis after training is complete', default='', metavar='ERROR_INFO_FP')
parser._action_groups.append(optional_args)

# load command line args
args = parser.parse_args(sys.argv[1:] if len(sys.argv) > 1 else ['-h'])
data_dir = args.d
config_fp = args.c
model_output_fp = args.m
verbose = args.v
error_info_fp = args.e

# set logger, start stopwatch
logging_level = logging.DEBUG if verbose else logging.INFO
logging.basicConfig(stream=sys.stdout, level=logging_level)
sw = utils.Stopwatch(); sw.start()

logging.info('Loading config...')
with open(config_fp, 'rt') as config_file:
    model_args = yaml.load(config_file)


logging.info('Creating new model...')

logging.info('Loading data...')
im_fps = glob2.glob(os.path.join(data_dir, '**/snippets/**/*.png'))
if not im_fps:
    logging.error('DATA_DIR \'%s\' contains no snippets in form \'snippets/*.png\'. Exiting...')
    exit() 
ims = utils.import_images(im_fps)
im_labels = utils.get_labels_from_fps(im_fps)


def build_model(model_args, X_train, y_train, X_test, y_test):
    logging.info('Building new model...')
    model = Model(**model_args)

    logging.info('Building BOVW...')
    model.BOVW_create(np.vstack((X_train, X_test)))

    logging.info('Training model...')
    model.train(X_train, y_train)
     
    predictions = model.predict(X_test)

    val_error  = utils.compute_error(y_test, predictions)
    num_total = len(y_test)
    num_incorrect = int(round(val_error * num_total))
    logging.info('Validation error: %g (%d/%d)' % (val_error, num_incorrect, num_total))
    return model, val_error

model_configs = hyperparameter_tuning.get_model_configs(model_args)
model, error_infos = hyperparameter_tuning.find_best_model(model_configs, build_model, ims, im_labels)

print(); logging.info('Saving model...')
utils.save_model(model, model_output_fp)
logging.info('Model saved to \'%s\'.' % model_output_fp)

if error_info_fp:
    logging.info('Saving model error info...')
    with open(error_info_fp, 'w+b') as error_info_file:
        pickle.dump(error_infos, error_info_file)
    logging.info('Error info saved to \'%s\'.' % error_info_fp)

sw.stop()
print(); logging.info('Model creation took %s.' % sw.format_str())

