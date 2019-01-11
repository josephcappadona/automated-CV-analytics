import model, glob
from descriptor_extractors import orb
from random import shuffle
from sys import argv


g = list(glob.iglob('../data/smash/*_snippets/*.png'))
shuffle(g)
im_fps = g[:100]


if len(argv) == 1:
    print('Creating new model...')
    m = model.Model(orb)
    
    m.BOVW_create(im_fps, k=128, show=False)
    m.SVM_train(im_fps)
    
elif len(argv) == 2:
    print('Loading model from file...')
    try:
        model_fp = argv[1]
        m = pickle.load(open(model_fp, 'rb'))
    except Exception as e:
        print('ERROR: Could not load model (%s) - %s' % (model_fp, e))
        exit()
else:
    print('USAGE:  python test_model.py [MODEL_PATH]\n')
    exit()


print('Predicting...')
t = g[100:105]
p = m.SVM_predict(t)
t_l = m.get_labels(t)
print(list(zip(p, t_l)))

import code; code.interact(local=locals())