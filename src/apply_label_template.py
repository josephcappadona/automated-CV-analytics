import sys
import utils
import glob
from label_template import load_label_template, adapt_label_template


usage = \
'''
USAGE:  python apply_label_template.py LABEL_TEMPLATE_FP IMG_DIR
'''

args = sys.argv

if len(args) != 3:
    print(usage)
    exit()

# build local variables from command line arguments
label_template_fp = args[1]
img_dir = args[2]

print('\n')
print('LABEL_TEMPLATE_FP=%s' % label_template_fp)
print('IMG_DIR=%s' % img_dir)
print('\n\n')


print('Loading label template \'%s\'...\n' % label_template_fp)
xml_elmt_tree = load_label_template(label_template_fp)

print('Finding images in \'%s\'...\n' % img_dir)
im_fps = glob.glob(img_dir + '/*.png')

print('Adapting template to %d images...\n' % len(im_fps))
for im_fp in im_fps:
    adapt_label_template(xml_elmt_tree, im_fp)
print('Done. New label files are in the same directory as the corresponding image files.\n')

