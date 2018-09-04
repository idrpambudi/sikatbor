import sys
import os
sys.path.append(os.getcwd())

import params
import model.util as utils

import cv2
import numpy as np
import pandas as pd
import random

from skimage import img_as_ubyte, exposure
from skimage.color import rgb2gray
from sklearn.cluster import KMeans
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, quickshift, slic
from scipy import ndimage

input_size = params.input_size
batch_size = params.batch_size
threshold = params.threshold
model = params.model_factory()
model.load_weights(filepath="weights/weights.hdf5")

def cal_scale(row,col, rec):
    r, c = rec,rec
    if col>row:
        r = row/col*rec
    if row>col:
        c = col/row*rec
    return int(np.round(r)),int(np.round(c))

def normalize_col_300(row,col):
    c = 300
    r = row/col*300
    return int(np.round(r)),int(np.round(c))

def rescale(row,col,inc):
    mx = np.max((row,col))+inc
    r,c = mx,mx
    if col>row:
        r = row/col*mx
    if row>col:
        c = col/row*mx
    return int(np.round(r)),int(np.round(c))

def hole_fill(img):
    # Copy the thresholded image.
    im_floodfill = img.copy()
    im_floodfill[:,0] = 0
    im_floodfill[0,:] = 0
    im_floodfill[:,-1] = 0
    im_floodfill[-1,:] = 0
     
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
     
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
     
    # Combine the two images to get the foreground.
    im_out = img | im_floodfill_inv
    return im_out
      
def improve_with_edge(img,mask,rect,fname):
    edges = cv2.Canny(img,100,200)

    # stretch the mask to produce new larger mask
    mask =  np.asarray(mask*255, dtype=np.uint8)
    
    kernel = np.ones((5,5),np.uint8)
    mask2 = cv2.dilate(mask,kernel,iterations = 1)
    new_mask = np.logical_and(mask2>0,edges)

    # fill hole to remove small error in outer region of laptop    
    # and create new improved mask
    edgeOrMask = (np.logical_or(new_mask,mask)*255).astype('uint8')

    nm3 = hole_fill(edgeOrMask)

    # create mask for grabcut
    mask = np.ones((nm3.shape[0],nm3.shape[1]),dtype='uint8')*2 
    utils.save_image('mask+edges-'+fname+'.png',img_as_ubyte(nm3))

    kernel = np.ones((25,25),np.uint8)
    erosion = 255-cv2.erode(255-nm3,kernel,iterations = 1)

    utils.save_image('erotion of background-'+fname+'.png',img_as_ubyte(erosion))

    mask[erosion == 0] = 0

    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(nm3,kernel,iterations = 1)

    utils.save_image('erotion of foreground-'+fname+'.png',img_as_ubyte(erosion))

    mask[erosion == 255] = 1
    utils.save_image('combination of erotion of foreground and background-'+fname+'.png',img_as_ubyte(mask*255))

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    mask, bgdModel, fgdModel = cv2.grabCut(img,mask,rect,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_MASK)
    utils.save_image('result grabcut + superpixel + edge-'+fname+'.png',img_as_ubyte(mask*255))
    nm3 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    mask = blur_mask(nm3)

    return mask

def blur_mask(mask):
    mask = np.asarray((mask>0)*255, dtype=np.uint8)
    mask = cv2.resize(mask, (0,0), fx=4.0, fy=4.0)
    kernel = np.ones((25,25),np.uint8)
    mask2 = cv2.dilate(mask,kernel,iterations = 1)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.blur(mask,(25,25))

    mask = cv2.resize(mask, (0,0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    return mask

def add_alpha(img,mask):
    new_img = np.zeros((img.shape[0],img.shape[1],4),dtype=np.uint8)
    new_img[:,:,1] = img[:,:,1]
    new_img[:,:,2] = img[:,:,2]
    new_img[:,:,0] = img[:,:,0]
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGBA)

    new_img[:,:,3] = mask
    return new_img

def clus_img(img, n_clusters=8):
    if len(img.shape)==2:
        nimg = np.zeros((img.shape[0],img.shape[1],3))
        nimg[:,:,0] = img[:,:,0]
        nimg[:,:,1] = img[:,:,1]
        nimg[:,:,2] = img[:,:,2]
        img = nimg
    img = img[:,:,:3]
    df_ = img.reshape([-1,3])
    arr = np.arange(len(df_))
    random.shuffle(arr)
    arr = arr[:8000]
    ndf = df_[arr]
    kmeans = KMeans(n_clusters=n_clusters, max_iter=50).fit(ndf)
    labels_ = kmeans.predict(df_)
    return labels_

def improve_wS(mask,fl):
    mask =  np.asarray(mask*255, dtype=np.uint8)
    kernel = np.ones((15,15),np.uint8)
    mask2 = cv2.dilate(mask.astype(np.uint8),kernel,iterations = 1) # bigger mask

    reg3 = np.logical_xor(mask2>0,mask).reshape([-1]) # new region is the difference of bigger and original mask

    reg1 = mask.reshape([-1]) # unet mask
    reg2 = np.logical_and(np.ones((mask.shape[0], mask.shape[1])), (1-(mask2>0))).reshape([-1]) # background region

    lreg3 = np.unique(fl[reg3]) # all unique segments
    for ii in lreg3:
        n1 = (np.logical_and((fl==ii), reg1)*1).sum() # more from unet mask
        n2 = (np.logical_and((fl==ii), reg2)*1).sum() # more in background
        n3 = np.logical_and((fl==ii), reg3)
        if (n1 > n2):# or n3.sum() > n2:
            reg1 = np.logical_or(n3,reg1)
    return reg1.reshape([mask.shape[0],mask.shape[1]])

def create_rect_from_unet_mask(mask):
    x,y = 0,0
    h,w = mask.shape[:2]
    h -= 1
    w -= 1
    while y+10<h-1 and mask[y+10,:].sum()==0 and y<h-1:
        y+=1
    while x+10<w-1 and mask[:,x+10].sum()==0 and x<w-1:
        x+=1
    while mask[h-10,:].sum()==0 and h>y+1:
        h-=1
    while mask[:,w-10].sum()==0 and w>x+1:
        w-=1
    return (x,y,w-x,h-y)

def removeSmallObject(mask):
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(\
        ((mask>0)*255).astype(np.uint8), connectivity=8)
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 10000

    #your answer image
    img2 = np.zeros((output.shape),dtype=np.uint8)
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 1
    
    mask = img2*mask
    return mask

def predict_and_save(img, fname, post_proc):
    if post_proc:
        return predict_and_save_post_proc(img, fname)
    else:
        return predict_and_save_no_post_proc(img, fname)

def predict_and_save_post_proc(img, fname):
    utils.save_image('ori-img-'+fname,img)
    x_batch = []
    row,col = img.shape[:2]
    ori_img = img.copy()

    row,col = normalize_col_300(row,col)
    img_300 = cv2.resize(img, (col,row))
    
    row,col = img_300.shape[:2]
    row,col = cal_scale(row,col,input_size)
    img = cv2.resize(img_300, (col,row))

    im = cv2.imread('input/IMG_20171019_175144_scaled.jpg')# np.zeros((input_size,input_size,3))
    im = cv2.resize(im, (input_size,input_size))
    im[:row,:col,:] = img
    utils.save_image('input-unet-'+fname,im)

    x_batch.append(im)
    x_batch = np.array(x_batch, np.float32) / 255
    preds = model.predict_on_batch(x_batch)
    preds = np.squeeze(preds, axis=3)
    pred = preds[0]
    utils.save_image('output-unet-'+fname+'.png',img_as_ubyte(pred))
    
    img = ori_img.copy()
    row,col = img.shape[:2]
    row,col = cal_scale(row,col,input_size)

    pred = pred[:row,:col]
    
    pred = cv2.resize(pred, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_CUBIC)
    
    # if row*col > img.shape[0]*img.shape[1] :
    #     pred = cv2.resize(pred, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_CUBIC)
    # else :
    #     img = cv2.resize(img, (col,row))

    # print(pred.shape, img.shape)

    mask = pred<0.5

    kernel = np.ones((3,3),np.uint8)
    mask = cv2.dilate((mask*255).astype(np.uint8),kernel,iterations = 1) # bigger mask
    mask = removeSmallObject(mask>0)
    
    rect = create_rect_from_unet_mask(mask)

    mask = improve_with_edge(img,mask,rect,fname)

    new_img = add_alpha(img,mask)

    cv2.imwrite('output/result/post_proc/'+fname+'.png', new_img)
    return mask

def predict_and_save_no_post_proc(img, fname):
    utils.save_image('ori-img-'+fname,img)
    x_batch = []
    row,col = img.shape[:2]
    ori_img = img.copy()

    row,col = normalize_col_300(row,col)
    img_300 = cv2.resize(img, (col,row))
    
    row,col = img_300.shape[:2]
    row,col = cal_scale(row,col,input_size)
    img = cv2.resize(img_300, (col,row))

    im = cv2.imread('input/IMG_20171019_175144_scaled.jpg')# np.zeros((input_size,input_size,3))
    im = cv2.resize(im, (input_size,input_size))
    im[:row,:col,:] = img
    utils.save_image('input-unet-'+fname,im)

    x_batch.append(im)
    x_batch = np.array(x_batch, np.float32) / 255
    preds = model.predict_on_batch(x_batch)
    preds = np.squeeze(preds, axis=3)
    pred = preds[0]
    utils.save_image('output-unet-'+fname+'.png',img_as_ubyte(pred))
    
    img = ori_img.copy()
    row,col = img.shape[:2]
    row,col = cal_scale(row,col,input_size)

    pred = pred[:row,:col]

    pred = cv2.resize(pred, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_CUBIC)
    
    # if row*col > img.shape[0]*img.shape[1] :
    #     pred = cv2.resize(pred, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_CUBIC)
    # else :
    #     img = cv2.resize(img, (col,row))

    # print(pred.shape, img.shape)

    mask = pred<0.5#fp.reshape([img.shape[0],img.shape[1]])
    mask2 = mask.copy()
    mask = removeSmallObject(mask>0)
    if mask.sum() == 0 :
        mask = mask2.copy()

    new_img = add_alpha(img,np.asarray(mask*255, dtype=np.uint8))

    cv2.imwrite('output/result/no_post_proc/'+fname+'.png', new_img)
    return mask
