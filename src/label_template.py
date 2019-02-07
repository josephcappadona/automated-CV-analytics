import xml.etree.ElementTree as ET
from PIL import Image
import os
# load template
# get images to apply template to
# for each image
#     check that size is the same
#     change template
#     save template

def load_label_template(label_template_fp):
    return ET.parse(label_template_fp)


def get_folder(filepath):
    return filepath.split('/')[-2]

def get_filename(filepath):
    return filepath.split('/')[-1]

def sizes_match(root, img_filepath):

    size_elmt = root.find('.//size')
    template_width = int(size_elmt.find('.//width').text)
    template_height = int(size_elmt.find('.//height').text)

    im = Image.open(img_filepath)
    img_width, img_height = im.size

    return (img_width == template_width) and (img_height == template_height)

def remove_extension(filepath):
    return '.'.join(filepath.split('.')[:-1])

def adapt_label_template(tree, img_filepath):

    root = tree.getroot()
    
    # if template size does not match image size, ask user to confirm template application
    if not sizes_match(root, img_filepath):
        print('Size of \'%s\' does not match given template. Adapt template anyway? (y/n)')

        ans = None
        while ans not in ['y', 'n']:
            ans = input().lower()
        
        if ans == 'n':
            return

    folder_elmt = root.find('.//folder')
    folder_elmt.text = get_folder(img_filepath)

    filename_elmt = root.find('.//filename')
    filename_elmt.text = get_filename(img_filepath)

    path_elmt = root.find('.//path')
    path_elmt.text = os.getcwd() + '/' + img_filepath

    new_label_fp = remove_extension(img_filepath) + '.xml'
    tree.write(new_label_fp)

