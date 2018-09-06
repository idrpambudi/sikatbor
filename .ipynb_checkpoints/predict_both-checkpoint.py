from skimage.io import imread 
from model.u_net_6 import predict_and_save, cal_scale
from additional_channel import used_additional_channel

import sys
import glob
import cv2
import params
import optparse
import numpy as np
import pandas as pd


folder_name = input('Result folder name: ')

def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = y_true.reshape([-1])>0
    y_pred_f = y_pred.reshape([-1])>0

    intersection = np.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    return score    

def false_negative(y_true, y_pred):
    y_true_f = y_true.reshape([-1])
    y_pred_f = y_pred.reshape([-1])

    fn = np.sum(y_true_f * (1-y_pred_f))
    score = fn / y_true_f.sum()
    return score, fn  

folder = 'input/valid/'
fol_ms = 'input/masks/'
lfn = glob.glob(folder+'*.*')
print(len(lfn))
ld = []
lf = []
lp = []
ld_post = []
lf_post = []
lp_post = []
num = len(lfn)
for fn in lfn[:num]:
    img = cv2.imread(fn, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = imread(fol_ms+fn[len(folder):]+'.gif')
    row,col = mask.shape[:2]
    row,col = cal_scale(row,col, params.input_size)
    
    for func in used_additional_channel():
        img = func(img)
    
    print(fn)
    new_mask = predict_and_save(img, fn[len(folder):], post_proc=True)>0
    ld_post.append(dice_coeff(mask, new_mask))
    fn_post, pixel_post = false_negative(mask,new_mask)
    lf_post.append(fn_post)
    lp_post.append(pixel_post)
    
    new_mask = predict_and_save(img, fn[len(folder):], post_proc=False)>0
    ld.append(dice_coeff(mask, new_mask))
    fn, pixel = false_negative(mask,new_mask)
    lf.append(fn)
    lp.append(pixel)


df_ = pd.DataFrame({'list filename': [fn[len(folder):] for fn in lfn[:num]],
              'dice': ld_post,'false_negative': lf_post, 'pixel-fn': lp_post})

with open('output/result/result.txt', 'w') as f:
    f.write("Post Processing Result\n")
    f.write(str(df_.mean()) + '\n\n')
    
df_.to_csv('output/hasil_post.csv', index=False)


df_ = pd.DataFrame({'list filename': [fn[len(folder):] for fn in lfn[:num]],
              'dice': ld,'false_negative': lf, 'pixel-fn': lp})
with open('output/result/result.txt', 'a') as f:
    f.write("No Post Processing Result\n")
    f.write(str(df_.mean()) + '\n\n')

df_.to_csv('output/hasil_no_post.csv', index=False)
