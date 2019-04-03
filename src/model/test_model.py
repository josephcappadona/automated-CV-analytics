import argparse
import pickle
import sys
import logging
import utils
import glob2
import os
from model import Model



parser = argparse.ArgumentParser(prog=('python %s' % sys.argv[0]), description='Test an image classification model.')
optional_args = parser._action_groups.pop()
required_args = parser.add_argument_group('required arguments')
required_args.add_argument('-d', type=str, help='The location of the testing snippets', required=True, metavar='DATA_DIR')
required_args.add_argument('-m', type=str, help='The location of the model to test', required=True, metavar='MODEL_FP')
optional_args.add_argument('-v', help='Verbose output (debug)', action='store_true')
parser._action_groups.append(optional_args)

args = parser.parse_args(sys.argv[1:] if len(sys.argv) > 1 else ['-h'])
data_dir = args.d
model_fp = args.m
logging_level = logging.DEBUG if args.v else logging.INFO
logging.basicConfig(stream=sys.stdout, level=logging_level)

logging.info('Loading model from file...')
with open(model_fp, 'rb') as model_file
    model = pickle.load(model_file)
    if type(model) == list:
        logging.error('\'%s\' contains list, not model. Exiting...' % model_fp)

logging.info('Loading data...')
im_fps = glob2.glob(os.path.join(data_dir, '**/snippets/**/*.png'))
ims = utils.import_images(im_fps)
im_labels = utils.get_labels_from_fps(im_fps)
    
logging.info('Testing model...')
predictions = model.predict(ims)

accuracy = 1 - utils.compute_error(im_labels, predictions)
num_total = len(im_labels)
num_correct = int(round(accuracy * num_total))
logging.info('Accuracy: %g (%d/%d)' % (accuracy, num_correct, num_total))
if accuracy != 1:
    logging.info('Incorrectly labeled images:')
    for i, (label, prediction) in enumerate(zip(im_labels, predictions)):
        if label != prediction:
            logging.info('img_fp=%s,  label=%s,  prediction=%s' % (im_fps[i], label, prediction))

