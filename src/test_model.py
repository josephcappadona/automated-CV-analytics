import sys
import utils
import glob
import model


args = utils.parse_args(sys.argv)

usage = "\nUSAGE:  python test_model.py -m MODEL_INPUT_FP -d DATA_DIR\n"
if not args['m'] or not args['d']: # if no DATA_DIR specified
    missing_arg = 'MODEL_INPUT_FP' if not args['m'] else 'DATA_DIR'
    print('No %s specified.\n' % missing_arg)
    print(usage)
    exit()


data_dir = args['d']
model_input_fp = args['m']

print('\n')
print('DATA_DIR=%s' % data_dir)
print('MODEL_INPUT_FP=\'%s\'' % model_input_fp)
print('\n\n')


im_fps = glob.glob(data_dir + '/*_snippets/*.png')

print('Loading model from file...\n')
m = model.Model.load_model(model_input_fp)

print('Loading data...\n')
ims = utils.import_images(im_fps)
im_labels = utils.get_labels_from_fps(im_fps)
    
print('Testing...\n')
y_hat = m.SVM_predict(ims)
print('Accuracy: %g\n' % utils.get_score(im_labels, y_hat))
for i, (y, y_) in enumerate(zip(im_labels, y_hat)):
    if y != y_:
        print(im_fps[i], (y, y_))
