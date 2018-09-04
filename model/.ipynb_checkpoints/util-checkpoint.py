import cv2
import numpy as np
from skimage.io import imsave

def cal_scale(row,col, rec):
    r, c = rec,rec
    if col>row:
        r = row/col*rec
    if row>col:
        c = col/row*rec
    return int(np.round(r)),int(np.round(c))

def save_image(fname, img):
    # print(fname)
    row,col = img.shape[:2]
    row,col = cal_scale(row,col,100)
    img = cv2.resize(img, (col,row), interpolation = cv2.INTER_CUBIC)
    imsave('output/process/'+fname,img)