import sys
import pathlib
import glob2
import xml.etree.ElementTree as ET
from PIL import Image
from snippets import create_snippet, create_negative_snippets


def get_label_fps(dir_):
    label_fps = glob2.iglob(dir_ + '/**/*.xml')
    return label_fps

def remove_extension(filename):
    return '.'.join(filename.split('.')[:-1])

def get_label_img_fp_pairs(dir_):

    label_img_fp_pairs = []
    label_fps = get_label_fps(dir_)

    for label_fp in label_fps:
        label_fp_base = remove_extension(label_fp)
        matching_img_fp = label_fp_base + '.png'
        
        if pathlib.Path(matching_img_fp).is_file(): # if image file exists
            label_img_fp_pairs.append((label_fp, matching_img_fp))

    return label_img_fp_pairs


def get_boxes(label_fp):

    tree = ET.parse(label_fp)
    xml_root = tree.getroot()

    boxes = {}
    for child in xml_root:
        if child.tag == 'object':
            for attrib in child:
                if attrib.tag == 'name':
                    name = attrib.text
                elif attrib.tag == 'bndbox':
                    xmin, ymin, xmax, ymax = [int(coord.text) for coord in attrib]
            boxes[name] = (xmin-1, ymin-1, xmax-1, ymax-1) # subtract 1 b/c LabelImg does not 0-index
    return boxes


if __name__ == '__main__':

    args = sys.argv

    if len(args) != 3:
        print('USAGE:  python create_text_snippets.py FRAMES_AND_LABELS_DIR\n\nExample:  python create_text_snippets.py output/my_video/frames')
        exit(1)
    data_dir = args[1]
    video_name = data_dir.split('/')[-2]

    label_img_fp_pairs = get_label_img_fp_pairs(data_dir)
    print('%d labeled images found.' % len(label_img_fp_pairs))

    print('Creating snippets...')
    count = 0
    for label_fp, img_fp in label_img_fp_pairs:
        print(label_fp, img_fp)

        boxes = get_boxes(label_fp)
        im = Image.open(img_fp)
        
        for box_name, box_coords in boxes.items():
            if create_snippet(video_name, im, box_name, box_coords):
                count += 1
        n_neg = create_negative_snippets(video_name, im, img_fp, boxes)
        count += n_neg
    print('%d snippets saved to \'output/%s/snippets\'' % (count, video_name))
