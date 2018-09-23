import pytesseract
from PIL import Image, ImageEnhance, ImageOps, ImageFilter

def image_to_text(im):

    new_size = tuple(3*x for x in im.size)
    im = im.resize(new_size, Image.ANTIALIAS)

    enhancer = ImageEnhance.Contrast(im)
    im = enhancer.enhance(2)
    im = im.convert('L')
    im = ImageOps.invert(im)
    im = im.point(lambda x: 0 if x < 100 else 255)
    im = im.filter(ImageFilter.SMOOTH_MORE)

    return pytesseract.image_to_string(im, config='-psm 6'), im


