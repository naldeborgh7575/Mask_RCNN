import cv2
from glob import glob
from os.path import join

def check_imgs(imgdir):
    imgs = glob(join(imgdir, '*.png')) + glob(join(imgdir, '*.jpg'))
    print('Checking {} images for correct shape'.format(str(len(imgs))))
    for img in imgs:
        im = cv2.imread(img)
        if im.shape != (256,256,3):
            print(img + 'shape: ' + str(im.shape))