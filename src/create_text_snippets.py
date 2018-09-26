from sys import argv
from subprocess import call
from os import listdir, mkdir
from os.path import isfile, join, exists
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
from image_utils import preprocess_image
import xml.etree.ElementTree as ET

if __name__ == '__main__':
    VIDEO_NAME = argv[1]

DATA_DIR = 'data'
IMAGE_DIR = DATA_DIR + '/' + VIDEO_NAME + '/frames'
LABEL_DIR = DATA_DIR + '/' + VIDEO_NAME + '/labels'
TEXT_SNIPPETS_DIR = DATA_DIR + '/' + VIDEO_NAME + '/text_snippets'


def get_image_file_paths():
    
    img_file_paths = [join(IMAGE_DIR, f) for f in listdir(IMAGE_DIR) if isfile(join(IMAGE_DIR, f))]
    return img_file_paths


def get_label_file_paths():
    
    label_file_paths = [join(LABEL_DIR, f) for f in listdir(LABEL_DIR) if isfile(join(LABEL_DIR, f)) and f.endswith('.xml')]

    return label_file_paths


def get_label_img_file_path_pairs():

    label_img_file_path_pairs = []
    label_file_paths = get_label_file_paths()

    for label_file_path in label_file_paths:
        label_file_name_base = label_file_path.split('/')[-1].split('.')[0]
        matching_img_file_path = IMAGE_DIR + '/' + label_file_name_base + '.png'
        
        label_img_file_path_pairs.append((label_file_path, matching_img_file_path))

    return label_img_file_path_pairs


def create_text_snippets(label_file_path, img_file_path, save=True, ret=False):
    tree = ET.parse(label_file_path)
    root = tree.getroot()

    boxes = {}
    for child in root:
        if child.tag == 'object':
            for attrib in child:
                if attrib.tag == 'name':
                    name = attrib.text
                elif attrib.tag == 'bndbox':
                    xmin, ymin, xmax, ymax = [int(coord.text) for coord in attrib]
            boxes[name] = (xmin, ymin, xmax, ymax)

    im = Image.open(img_file_path)
    img_file_base = img_file_path.split('/')[-1].split('.')[0]

    box_imgs = []
    for box_name, box_coords in boxes.iteritems():
        box_img = im.copy().crop(box_coords)
        box_img = preprocess_image(box_img)
        box_imgs.append(box_img)
        if save:
            if not exists(TEXT_SNIPPETS_DIR):
                mkdir(TEXT_SNIPPETS_DIR)
            box_img.save(TEXT_SNIPPETS_DIR + '/' + img_file_base + '_' + box_name + '.tif')
            box_file = open(TEXT_SNIPPETS_DIR + '/' + img_file_base + '_' + box_name + '.box', 'w+')
            box_file.close()

    if ret:
        return box_imgs


for label_fp, img_fp in get_label_img_file_path_pairs():
    print label_fp, img_fp
    create_text_snippets(label_fp, img_fp)

