import pytesseract
from image_utils import image_preprocess

def image_to_text(im):
    
    im = preprocess_image(im)

    return pytesseract.image_to_string(im, lang='eng', config='-psm 6')


