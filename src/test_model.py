import sys
import utils
import glob2
import model


args = utils.parse_args(sys.argv)

usage = "\nUSAGE:  python test_model.py -d DATA_DIR -t DECISION_MODEL_TYPE -m MODEL_INPUT_FP\n"
if not args['m'] or not args['t'] or not args['d']: # if no DATA_DIR specified
    missing_arg = 'MODEL_INPUT_FP' if not args['m'] \
                      else 'DECISION_MODEL_TYPE' if not args['t'] \
                      else 'DATA_DIR'
    print('No %s specified.\n' % missing_arg)
    print(usage)
    exit()


data_dir = args['d']
model_type = args['t']
model_input_fp = args['m']

print('\n')
print('DATA_DIR=%s' % data_dir)
print('MODEL_INPUT_FP=\'%s\'' % model_input_fp)
print('\n\n')


im_fps = glob2.glob(data_dir + '/**/snippets/**/*.png')

print('Loading model from file...\n')
m = model.Model.load_model(model_input_fp)

print('Loading data...\n')
ims = utils.import_images(im_fps)
im_labels = utils.get_labels_from_fps(im_fps)
    
print('Testing %s model...\n' % model_type.upper())
y_hat = m.predict(model_type, ims)
print('Accuracy: %g\n' % utils.get_score(im_labels, y_hat))
for i, (y, y_) in enumerate(zip(im_labels, y_hat)):
    if y != y_:
        print(im_fps[i], (y, y_))
