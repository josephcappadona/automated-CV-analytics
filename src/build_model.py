import sys
import utils
import time
import glob
import descriptor_extractors
import model
import os
import warnings; warnings.filterwarnings('ignore')


args = utils.parse_args(sys.argv)

usage = "\nUSAGE:  python test_model.py -d DATA_DIR [-o MODEL_OUTPUT_FP] [--consider_descriptors 0/1] [--consider_colors 0/1]\n\nDefault values if argument not specified:\n-o \"output/model %Y-%m-%d %H:%M:%S.pkl\"\n--consider_descriptors 1\n--consider_colors 1\n"
if not args['d']: # if no DATA_DIR specified
    print('No DATA_DIR specified.\n')
    print(usage)
    exit()


data_dir = args['d']
model_output_fp = args['o'] if args['o'] \
                      else ('output/model %s.pkl' %
                            time.strftime('%Y-%m-%d %H:%M:%S',
                                          time.localtime()))
consider_descriptors = bool(int(args['consider_descriptors'])) \
                           if args['consider_descriptors'] else True
consider_colors = bool(int(args['consider_colors'])) \
                      if args['consider_colors'] else True

print('\n')
print('DATA_DIR=%s' % data_dir)
print('MODEL_OUTPUT_FP=\'%s\'' % model_output_fp)
print('consider_descriptors=%s' % consider_descriptors)
print('consider_colors=%s' % consider_colors)
print('\n\n')


print('Creating new model...\n')
im_fps = glob.glob(data_dir + '/*_snippets/*.png')
if not im_fps:
    print('ERROR:  DATA_DIR \'%s\' contains no snippets in form \'*_snippets/*.png\'.\n')
    exit()
 
print('Loading data...')
ims = utils.import_images(im_fps)
im_labels = utils.get_labels_from_fps(im_fps)

print('Building model...\n')
m = model.Model(descriptor_extractors.orb_create)

if consider_descriptors:
    print('Building BOVW...')
    m.BOVW_create(ims, k=[8, 16, 32, 64, 128], show=False)

print('Training SVM...')
m.SVM_train(ims,
            im_labels,
            consider_descriptors=consider_descriptors,
            consider_colors=consider_colors)

print('Saving model...')
model_output_dir = utils.get_directory(model_output_fp)
os.makedirs(model_output_dir, exist_ok=True)
m.save_model(model_output_fp)
print('Model saved to \'%s\'.\n' % model_output_fp)

