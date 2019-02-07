import sys
import utils
import glob
from label_template import load_label_template, adapt_label_template

args = utils.parse_args(sys.argv)

usage = '\nUSAGE:  python apply_label_template.py -t LABEL_TEMPLATE_FP -f IMG_DIR\n'
if not args['t'] and not args['f']:
    print(usage)
    exit()

label_template_fp = args['t']
img_dir = args['f']

print('\n')
print('LABEL_TEMPLATE_FP=%s' % label_template_fp)
print('IMG_DIR=%s' % img_dir)
print('\n\n')


print('Loading label template \'%s\'...\n' % label_template_fp)
elmt_tree = load_label_template(label_template_fp)

print('Finding images in \'%s\'...\n' % img_dir)
im_fps = glob.glob(img_dir + '/*.png')

print('Adapting template to %d images...\n' % len(im_fps))
for im_fp in im_fps:
    adapt_label_template(elmt_tree, im_fp)
print('Done.\n')

