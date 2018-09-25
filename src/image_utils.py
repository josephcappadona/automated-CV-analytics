from PIL import Image, ImageEnhance, ImageOps, ImageFilter, ImageChops


# taken from https://gist.github.com/mattjmorrison/932345
def trim(im, border_color):
    bg = Image.new(im.mode, im.size, border_color)
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


def preprocess_image(im):
    new_size = tuple(4*x for x in im.size)
    im = im.resize(new_size, Image.ANTIALIAS)

    enhancer = ImageEnhance.Contrast(im)
    im = enhancer.enhance(2)
    im = im.convert('L')
    im = ImageOps.invert(im)
    im = im.point(lambda x: 0 if x < 75 else 255)
    im = im.filter(ImageFilter.SMOOTH_MORE)
    im = trim(im, 255)

    return im

