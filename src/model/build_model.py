import argparse
import sys
import yaml
import logging
import utils
import time
import glob2
import descriptor_extractors
import os
import warnings; warnings.filterwarnings('ignore') # TODO: filter only warnings I want to filter
from model import Model


parser = argparse.ArgumentParser(prog=('python %s' % sys.argv[0]), description='Build an image classification model.')
optional_args = parser._action_groups.pop()
required_args = parser.add_argument_group('required arguments')
required_args.add_argument('-d', type=str, help='The location of the training snippets', required=True, metavar='DATA_DIR')
required_args.add_argument('-c', type=str, help='The location of the config file with the model parameters', required=True, metavar='CONFIG')
required_args.add_argument('-m', type=str, help='The output location of the created model', required=True, metavar='MODEL_OUTPUT_FP')
optional_args.add_argument('-v', help='Verbose output (debug)', action='store_true')
parser._action_groups.append(optional_args)

args = parser.parse_args(sys.argv[1:] if len(sys.argv) > 1 else ['-h'])
data_dir = args.d
config_fp = args.c
model_output_fp = args.m
logging_level = logging.DEBUG if args.v else logging.INFO
logging.basicConfig(stream=sys.stdout, level=logging_level)

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

logging.info('Building new model...')
model = Model(**model_args)

logging.info('Building BOVW...')
model.BOVW_create(ims)

logging.info('Training model...')
model.train(ims, im_labels)
 
logging.info('Computing validation error...')
predictions = model.predict(ims)

validation_error  = utils.compute_error(im_labels, predictions)
num_total = len(im_labels)
num_incorrect = int(round(validation_error * num_total))
logging.info('Validation error: %g (%d/%d)' % (validation_error, num_incorrect, num_total))

logging.info('Saving model...')
utils.save_model(model, model_output_fp)
logging.info('Model saved to \'%s\'.' % model_output_fp)

