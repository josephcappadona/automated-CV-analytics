from sys import argv as args
import utils
import glob2
import os
from model import Model


usage = \
'''
USAGE:  python test_model.py MODEL_INPUT_FP DATA_DIR
'''
if len(args) == 1:
    print(usage)
    exit()

try:
    # build local variables from command line arguments
    model_input_fp = args[1]
    data_dir = args[2]

except IndexError:
    # if not MODEL_INPUT_FP or DATA_DIR specified
    print('\nERROR:  Command line arguments incomplete.')
    print(usage)
    exit()


print('\n')
print('DATA_DIR=%s' % data_dir)
print('MODEL_INPUT_FP=\'%s\'' % model_input_fp)
print('\n\n')


print('Loading model from file...\n')
model = Model.load(model_input_fp)

print('Loading data...\n')
im_fps = glob2.glob(os.path.join(data_dir, '**/snippets/**/*.png'))
ims = utils.import_images(im_fps)
im_labels = utils.get_labels_from_fps(im_fps)
    
print('Testing BOVW+%s model...\n' % model.model_type)
predictions = model.predict(ims)

num_correct = sum(1 if l == p else 0 for (l, p) in zip(im_labels, predictions))
num_total = len(im_labels)
accuracy = num_correct / num_total


print('Accuracy: %g (%d/%d)\n' % (accuracy, num_correct, num_total))
print('Incorrectly labeled images:\n')
for i, (label, prediction) in enumerate(zip(im_labels, predictions)):
    if label != prediction:
        print(im_fps[i])
        print('\tCorrect label: %s' % label)
        print('\tPredicted label: %s' % prediction)
        print('')

