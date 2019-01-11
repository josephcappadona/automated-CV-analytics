from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageChops


# crop image to remove empty space around text
# taken from https://gist.github.com/mattjmorrison/932345
def trim(im, border_color):
    bg = Image.new(im.mode, im.size, border_color)
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


# currently configured specifically for ACR text
def preprocess_image(im):
    
    new_size = tuple(4*x for x in im.size)
    im = im.resize(new_size, Image.BICUBIC)

    im = im.convert('L') # make monochrome
    im = ImageOps.invert(im) # make text background white and text black
    im = im.point(lambda x: 0 if x < 100 else 255) # binarize

    # invert image if background is black
    if im.getpixel((0,0)) < 80:
        im = ImageOps.invert(im)
    im = trim(im, im.getpixel((0,0)))
 
    return im

