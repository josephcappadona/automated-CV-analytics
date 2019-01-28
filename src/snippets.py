from random import randint
import numpy as np


def create_snippet(video_name, im, box_name, box_coords, output_dir='output'):

    box_img = im.copy().crop(box_coords)

    snippets_dir = output_dir + '/' + video_name + '/' + box_name + '_snippets'
    makedirs(snippets_dir, exist_ok=True)

    snippet_id = str(randint(10**7, 10**10))
    snippet_fp = snippets_dir + '/' + box_name + '.' + snippet_id + '.png'
    box_img.save(snippet_fp)
    return True

def create_negative_snippets(video_name, im, im_fp, boxes, output_dir='output'):

    neg_snippets_dir = output_dir + '/' + video_name + '/NEGATIVE_snippets'
    makedirs(neg_snippets_dir, exist_ok=True)

    occupied = np.zeros(im.size, dtype=np.uint8)
    for box_coords in boxes.values():
        x_min, y_min, x_max, y_max = box_coords
        occupied[x_min:x_max+1, y_min:y_max+1] = 1 # mark off areas containing label boxes

    w_im, h_im = im.size
    count = 0
    for box_name, box_coords in boxes.items():
        # (try to) make one negative box for each label box, try to make them the same size
        
        x_min, y_min, x_max, y_max = box_coords
        w_box, h_box = (x_max - x_min), (y_max - y_min)

        N = 10
        for i in range(N):
            # try at most 10 times to find a viable negative image somewhere random in the image
            # reduce the negative image box size slightly each iteration to make it more likely to find a negative
            x_rand = randint(0, w_im - w_box)
            y_rand = randint(0, h_im - h_box)

            # if new box overlaps with any label box (i.e, there is a 1 present in the array)
            if occupied[x_rand:x_rand+w_box, y_rand:y_rand+h_box].any():
                
                # reduce box size and try again
                w_box = int(w_box * 0.9)
                h_box = int(h_box * 0.9)
                if i == N-1:
                    print('\tCould not find negative box for label \'%s\' in image \'%s\'.' % (box_name, img_fp))
                continue

            # found good box!
            neg_box_coords = (x_rand, y_rand, x_rand + w_box, y_rand + h_box)
            create_snippet(video_name, im, 'NEGATIVE', neg_box_coords)
            count += 1
            break
    return count

