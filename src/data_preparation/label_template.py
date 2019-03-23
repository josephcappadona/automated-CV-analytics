import xml.etree.ElementTree as ET
from PIL import Image
import os
import utils


def load_label_template(label_template_fp):
    return ET.parse(label_template_fp)

def sizes_match(root, img_filepath):

    size_elmt = root.find('.//size')
    template_width = int(size_elmt.find('.//width').text)
    template_height = int(size_elmt.find('.//height').text)

    im = Image.open(img_filepath)
    img_width, img_height = im.size

    return (img_width == template_width) and (img_height == template_height)

def adapt_label_template(tree, img_filepath):

    root = tree.getroot()
    
    # if template size does not match image size, ask user to confirm template application
    if not sizes_match(root, img_filepath):
        print('Size of image \'%s\' does not match given template. Adapt template anyway? (y/n)')

        ans = None
        while ans not in ['y', 'n']:
            ans = input().lower()
        
        if ans == 'n':
            return

    # edit image parent folder
    folder_elmt = root.find('.//folder')
    folder_elmt.text = utils.get_parent_folder(img_filepath)

    # edit image filename
    filename_elmt = root.find('.//filename')
    filename_elmt.text = utils.get_filename(img_filepath)

    # edit image absolute filepath
    path_elmt = root.find('.//path')
    path_elmt.text = os.path.join(os.getcwd(), img_filepath)

    # write new label .xml file
    new_label_fp = utils.remove_extension(img_filepath) + '.xml'
    tree.write(new_label_fp)

