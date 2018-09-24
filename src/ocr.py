import pytesseract
from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageChops

def image_to_text(im):

    new_size = tuple(4*x for x in im.size)
    im = im.resize(new_size, Image.ANTIALIAS)

    #im = im.filter(ImageFilter.MedianFilter())
    enhancer = ImageEnhance.Contrast(im)
    im = enhancer.enhance(2)
    im = im.convert('L')
    im = ImageOps.invert(im)
    im = im.point(lambda x: 0 if x < 100 else 255)
    im = im.filter(ImageFilter.SMOOTH_MORE)
    im = trim(im, 255)

    return pytesseract.image_to_string(im, lang='eng', config='-psm 6')


# taken from https://gist.github.com/mattjmorrison/932345
def trim(im, border):
    bg = Image.new(im.mode, im.size, border)
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
