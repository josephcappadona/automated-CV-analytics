# Tesseract Font Trainer
# Adapted from https://github.com/this-is-ari/python-tesseract-3.02-training/blob/master/tesseract-trainer.py

from os import getuid
from sys import argv
if getuid() != 0:
    print 'ERROR: Script must be run as root!\n'
    exit(0)
if len(argv) != 3:
    print 'USAGE: sudo python VIDEO_NAME FONT_NAME\n'

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('[%(levelname)s] %(name)s: %(message)s\n')
handler.setFormatter(formatter)
logger.addHandler(handler)


VIDEO_NAME = argv[1]
FONT_NAME = argv[2]

LANG = "eng"
DATA_DIR = "data"
IMG_DIR = "%s/%s/text_snippets" % (DATA_DIR, VIDEO_NAME)
BOX_DIR = "%s/%s/box_files" % (DATA_DIR, VIDEO_NAME)
TR_DIR = "%s/%s/tr_files" % (DATA_DIR, VIDEO_NAME)

from os import mkdir
from os.path import exists
if not exists(TR_DIR):
    mkdir(TR_DIR)


from os import listdir
from os.path import join
from subprocess import call

logger.info('Running tesseract on image files in %s', IMG_DIR)
valid_img_extensions = ('.tif', '.tiff')
count = 0
for img_file in listdir(IMG_DIR):
    
    if img_file.endswith(valid_img_extensions):

        args = {'img_dir':IMG_DIR, 'img_file':img_file, 'tr_dir':TR_DIR,
                'lang':LANG, 'font_name':FONT_NAME, 'num':str(count)}
        command = "tesseract {img_dir}/{img_file} {tr_dir}/{lang}.{font_name}.exp{num} --psm 6 nobatch box.train.stderr".format(**args)

        print command
        call(command, shell=True)
        count += 1
if count == 0:
    logger.error('No valid image files found. Files must be of type TIFF.')
    exit(0)
logger.info('Processed %s image files, output .tr files to %s', str(count), TR_DIR)

tr_files = []
for tr_file in listdir(TR_DIR):
    if tr_file.endswith('.tr'):
        tr_files.append(join(TR_DIR, tr_file))
tr_file_list = ' '.join(tr_files)

box_files = []
for box_file in listdir(BOX_DIR):
    if box_file.endswith('.box'):
        box_files.append(join(BOX_DIR, box_file))
box_file_list = ' '.join(box_files)


logger.info('Building unicharset file in %s', TR_DIR)
unicharset_file_name = join(TR_DIR, 'unicharset')
command = 'unicharset_extractor --output_unicharset %s %s' % (unicharset_file_name, box_file_list)
call(command, shell=True)


logger.info('Building font_properties file in %s', TR_DIR)
font_properties = '%s 0 0 0 0 0' % FONT_NAME
font_properties_file_name = join(TR_DIR, 'font_properties')
with open(font_properties_file_name, 'w+') as font_properties_file:
    font_properties_file.write(font_properties)


command = 'shapeclustering -F %s -U %s -D %s %s' % (font_properties_file_name, unicharset_file_name, TR_DIR, tr_file_list)
print command
call(command, shell=True)


command = 'mftraining -F {} -U {} -O {}/{}.charset -D {} {}'.format(font_properties_file_name, unicharset_file_name, TR_DIR, FONT_NAME, TR_DIR, tr_file_list)
print command
call(command, shell=True)


command = 'cntraining -D %s %s' % (TR_DIR, tr_file_list)
print command
call(command, shell=True)


logger.info('Renaming necessary files')
call('mv %s/unicharset %s/%s.unicharset' % (TR_DIR, TR_DIR, FONT_NAME), shell=True)
call('mv %s/shapetable %s/%s.shapetable' % (TR_DIR, TR_DIR, FONT_NAME), shell=True)
call('mv %s/normproto %s/%s.normproto' % (TR_DIR, TR_DIR, FONT_NAME), shell=True)
call('mv %s/pffmtable %s/%s.pffmtable' % (TR_DIR, TR_DIR, FONT_NAME), shell=True)
call('mv %s/inttemp %s/%s.inttemp' % (TR_DIR, TR_DIR, FONT_NAME), shell=True)

logger.info('Combining tessdata')
command = 'combine_tessdata %s.' % join(TR_DIR, FONT_NAME)
call(command, shell=True)


logger.info('Copying training output to tessdata folder')
command = 'cp -f %s.* /usr/local/share/tessdata' % join(TR_DIR, FONT_NAME)
call(command, shell=True)

logger.info('Finished')

