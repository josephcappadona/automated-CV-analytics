import sys
import utils
import time
import glob2
import descriptor_extractors
import os
import warnings; warnings.filterwarnings('ignore') # TODO: filter only warnings I want to filter
from model import Model


usage = 'USAGE:  python build_model.py --data_dir DATA_DIR [--config CONFIG.yaml]'
if len(sys.argv) == 1:
    print(usage)
    exit()

args = utils.parse_args(sys.argv) # type(args) == defaultdict(str)

# build local variables from command line arguments
data_dir = args['data_dir']

if 'config' in args:
    import yaml
    with open(args['config'], 'rt') as config_file:
        model_args = yaml.load(config_file)
        for arg_name, arg in model_args.items():
            vars()[arg_name] = arg

if 'model_type' not in vars():
    model_type = 'SVM'
    model_params = {'kernel': 'linear'}

if 'model_output_fp' not in vars():
    model_output_fp = 'output/model %s.pkl' % \
                          time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

if 'consider_descriptors' not in vars():
    consider_descriptors = 1

if 'consider_colors' not in vars():
    consider_colors = 1

if 'data_transform' not in vars() or data_transform == 'None':
    data_transform = None
    data_transform_params = {}

if 'feature_selection' not in vars() or feature_selection == 'None':
    feature_selection = None
    feature_selection_params = {}

if 'approximation_kernel' not in vars() or approximation_kernel == 'None':
    approximation_kernel = None
    approximation_kernel_params = {}

# if no DATA_DIR specified
if not data_dir:
    print('No DATA_DIR specified.\n')
    print(usage)
    exit()


print('\n')
print('DATA_DIR=%s' % data_dir)
print('MODEL_OUTPUT_FP=\'%s\'' % model_output_fp)
print('model_type=%s' % model_type)
print('model_params=%s' % model_params)
print('consider_descriptors=%s' % consider_descriptors)
print('consider_colors=%s' % consider_colors)
print('data_transform=%s' % data_transform)
print('data_transform_params=%s' % data_transform_params)
print('feature_selection=%s' % feature_selection)
print('feature_selection_params=%s' % feature_selection_params)
print('approximation_kernel_params=%s' % approximation_kernel_params)
print('\n\n')


print('Creating new model...\n')

print('Loading data...')
im_fps = glob2.glob(os.path.join(data_dir, '**/snippets/**/*.png'))
if not im_fps:
    print('ERROR:  DATA_DIR \'%s\' contains no snippets in form \'snippets/*.png\'.\n')
    exit() 
ims = utils.import_images(im_fps)
im_labels = utils.get_labels_from_fps(im_fps)

print('Building new model...\n')
model = Model(descriptor_extractors.orb_create) # TODO: allow custom descriptor extractor

if consider_descriptors:
    print('Building BOVW...\n')
    model.BOVW_create(ims, k=[64], show=False) # TODO: allow custom cluster sizes

print('Training model...\n')
model.train(model_type, model_params,
            ims, im_labels,
            consider_descriptors, consider_colors,
            data_transform, data_transform_params,
            feature_selection, feature_selection_params,
            approximation_kernel, approximation_kernel_params)

print('Computing validation error...\n')
predictions = model.predict(ims)

num_incorrect = sum(1 if l != p else 0 for (l, p) in zip(im_labels, predictions))
num_total = len(im_labels)
validation_error = num_incorrect / num_total
print('Validation error: %g (%d/%d)\n\n' % (validation_error, num_incorrect, num_total))

print('Saving model...')
model_output_dir = utils.get_directory(model_output_fp)
os.makedirs(model_output_dir, exist_ok=True)
model.save(model_output_fp)
print('Model saved to \'%s\'.\n' % model_output_fp)

