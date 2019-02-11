from random import randint
from os import makedirs
import numpy as np


def create_snippet(video_name, im, box_name, box_coords, output_dir='output'):

    box_img = im.copy().crop(box_coords)

    snippets_dir = output_dir + '/' + video_name + '/snippets/' + box_name
    makedirs(snippets_dir, exist_ok=True)

    snippet_id = str(randint(10**7, 10**10))
    snippet_fp = snippets_dir + '/' + box_name + '.' + snippet_id + '.png'
    box_img.save(snippet_fp)
    return True

def is_blank(im):
    # TODO: try to simplify using image_utils.trim
    im = np.array(im)
    top_left_pixel = im[0, 0]
    is_not_blank = im[im != top_left_pixel].any()
    return (not is_not_blank)

def create_negative_snippets(video_name, im, im_fp, boxes, output_dir='output', neg_to_pos_ratio=1, verbose=False):

    # mark off areas containing label boxes
    occupied = np.zeros(im.size[::-1], dtype=np.uint8)
    for box_coords_list in boxes.values():
        for box_coords in box_coords_list:
            x_min, y_min, x_max, y_max = box_coords
            occupied[y_min:y_max+1, x_min:x_max+1] = 1

    w_im, h_im = im.size
    count = 0
    for box_name, box_coords_list in boxes.items():
        for box_coords in box_coords_list:
            for k in range(neg_to_pos_ratio):
                # (try to) make one negative snippet for each positive snippet
                
                x_min, y_min, x_max, y_max = box_coords
                w_box, h_box = (x_max - x_min), (y_max - y_min)

                N = 10
                for i in range(N):
                    # try at most N times to find a viable negative image somewhere random in the image
                    x_rand = randint(0, w_im - w_box)
                    y_rand = randint(0, h_im - h_box)
                    x_rand_min, x_rand_max = x_rand, x_rand + w_box
                    y_rand_min, y_rand_max = y_rand, y_rand + h_box

                    # if random bbox overlaps with any label bbox
                    # or subimage is blank
                    # then try to find new random bbox
                    does_overlap_label_bbox = occupied[y_rand_min:y_rand_max,
                                                       x_rand_min:x_rand_max].any()
                    subimage = im.crop((x_rand_min, y_rand_min, x_rand_max, y_rand_max))
                    if does_overlap_label_bbox or is_blank(subimage):
                        
                        if i == N-1 and verbose:
                            print('\tCould not find negative box for label \'%s\' in image \'%s\'.\n' % (box_name, im_fp))
                        continue

                    # found good box!
                    neg_box_coords = (x_rand, y_rand, x_rand + w_box, y_rand + h_box)
                    create_snippet(video_name, im, 'NEGATIVE', neg_box_coords)
                    count += 1
                    break
    return count

