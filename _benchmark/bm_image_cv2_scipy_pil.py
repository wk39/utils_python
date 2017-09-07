#import timeit
import time
from collections import OrderedDict

import cv2      # opencv
import PIL    # pillow (PIL)
import scipy.misc as scm    # scipy

import numpy as np

def load_image_cv2(fname):
    img = cv2.imread(fname,cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def load_image_pil(fname):
    return np.asarray(PIL.Image.open(fname))

def load_image_scipy(fname):
    return scm.imread(fname)    # actually scipy.misc.imread uses PIL

def bm_load_image():

    '''benchmark image loading'''

    fns = OrderedDict()
    fns['cv2'  ] = load_image_cv2
    fns['PIL'  ] = load_image_pil
    fns['scipy'] = load_image_scipy

    fname = 'cute-golden-retriever.jpg'

    for name, fn in fns.items():
        t0 = time.time()
        for i in range(20):
            fn(fname)
        dt = time.time()-t0
        print('%5s %.3f s'%(name,dt))


if __name__ == '__main__':

    print('warming up... ', end='')
    for i in range(10**5):
        j = i*1000
    print('done')


    print(bm_load_image.__doc__)
    bm_load_image()




