import sys
import pathlib
import os
import glob2
import utils
import xml.etree.ElementTree as ET
from PIL import Image
from snippets import create_snippet, create_negative_snippets
from collections import defaultdict


def get_label_fps(dir_):
    # find XML files output from labelImg
    label_fps = glob2.iglob(os.path.join(dir_, '**/*.xml'))
    return label_fps

def get_label_img_fp_pairs(dir_):

    label_img_fp_pairs = []
    label_fps = get_label_fps(dir_)

    for label_fp in label_fps:
        label_fp_base = utils.remove_extension(label_fp)
        matching_img_fp = label_fp_base + '.png'
        
        # if image file exists (which it should if there is a corresponding .xml file)
        if pathlib.Path(matching_img_fp).is_file():
            label_img_fp_pairs.append((label_fp, matching_img_fp))

    return label_img_fp_pairs


def get_bboxes(label_fp):

    tree = ET.parse(label_fp)
    xml_root = tree.getroot()

    # parse XML and extract bounding box coordinates and labels
    # type(bboxes) = defaultdict(list) in case multiple bboxes have the same label
    bboxes = defaultdict(list)
    for child in xml_root:
        if child.tag == 'object':
            for attrib in child:
                if attrib.tag == 'name':
                    name = attrib.text
                elif attrib.tag == 'bndbox':
                    # subtract 1 from each coord b/c LabelImg does not 0-index
                    xmin, ymin, xmax, ymax = [int(coord.text)-1 for coord in attrib]
            bboxes[name].append((xmin, ymin, xmax, ymax))
    return bboxes


usage = \
'''
USAGE:  python create_text_snippets.py FRAMES_AND_LABELS_DIR MEDIA_NAME [NEG_TO_POS_RATIO]

If arg is not specified:
NEG_TO_POS_RATIO=1

Video example:  python create_snippets.py output/my_video/frames my_video 2
Image example:  python create_snippets.py data/my_image my_image 2
'''

args = sys.argv

if len(args) < 3 or len(args) > 4:
    print(usage)
    exit(1)

# build local variables from command line arguments
data_dir = args[1]
video_name = args[2]
neg_to_pos_ratio = int(args[3]) if len(args) == 4 else 1

print('\n')
print('DATA_DIR=%s' % data_dir)
print('VIDEO_NAME=%s' % video_name)
print('NEG_TO_POS_RATIO=%d' % neg_to_pos_ratio)
print('\n\n')


print('Finding labeled images...')
label_img_fp_pairs = get_label_img_fp_pairs(data_dir)
print('%d labeled images found.\n' % len(label_img_fp_pairs))

print('Creating snippets...')
pos_snippet_count = 0
neg_snippet_count = 0
for label_fp, img_fp in label_img_fp_pairs:

    bboxes = get_bboxes(label_fp)
    im = Image.open(img_fp)
    
    for bbox_label, bbox_coords_list in bboxes.items():
        for bbox_coords in bbox_coords_list:
            if create_snippet(video_name, im, bbox_label, bbox_coords):
                pos_snippet_count += 1
    n_neg = create_negative_snippets(video_name,
                                     im,
                                     img_fp,
                                     bboxes,
                                     neg_to_pos_ratio=neg_to_pos_ratio)
    neg_snippet_count += n_neg
num_snippets = pos_snippet_count + neg_snippet_count
print('%d snippets (%d pos, %d neg) saved to \'output/%s/snippets\'' \
          % (num_snippets, pos_snippet_count, neg_snippet_count, video_name))

