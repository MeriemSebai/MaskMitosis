#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generation of the masks of the full training images
"""
import os
import pickle
import numpy as np
import math
from PIL import Image
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from shutil import copyfile
from distutils.dir_util import copy_tree
from scipy.ndimage.measurements import label


src='./output/masks' # folder that contains all mitosis ground truth masks as np arrays
dst='gt_train' # create an intermediate folder
if not os.path.exists(dst):
    os.makedirs(dst)
objects = []
fid=open('./segmentation_train.txt','r')
for filename in fid:    
    filename=filename.strip()    
    f = np.loadtxt(os.path.join(src,filename))
    mask=np.zeros([f.shape[0],f.shape[1]])
    labeled_array, num_features = label(f)
    
    if num_features>0:
        indices = [np.nonzero(labeled_array == k) for k in np.unique(labeled_array)[1:]]
        lens=[ind[0].size for ind in indices]
        max_len=np.max(lens)
        max_ind=np.argmax(lens)         
        for x,y in zip(indices[max_ind][0],indices[max_ind][1]):
            mask[x,y]=1
    
    obj_struct = {}
    obj_struct['imagename'] = ['{}_{}'.format(filename.split('_')[0],filename.split('_')[1])]    
    obj_struct['det'] = mask
    src_f=filename.split('.')[0]
    obj_struct['coord'] = [int(src_f.split('_')[2]),int(src_f.split('_')[3]),int(src_f.split('_')[4]),int(src_f.split('_')[5])]
    objects.append(obj_struct)
  
imagesetfile = 'train.txt'
with open(imagesetfile, 'r') as f:
    lines = f.readlines()
imagenames = [x.strip().split('/')[2] for x in lines]

# create the segmented mask of the full image 
for imagename in imagenames:
    objet = [obj for obj in objects if obj['imagename'][0]==imagename]   
    mask=np.zeros([1376,1539])       
    for obj in objet:
        x1=obj['coord'][0]
        y1=obj['coord'][1]
        x2=obj['coord'][2]
        y2=obj['coord'][3]  
        m=obj['det']        
        mask[y1:y2,x1:x2]=m         
    plt.imsave(os.path.join(dst,'{}.jpg'.format(imagename)), mask, cmap=cm.gray)
    
src='./gt_train' 
dst='/path/to/dataset/mitosis2014/gtImg'  
if not os.path.exists(dst):
    os.makedirs(dst) 
for f in os.listdir(src):    
    src_file=os.path.join(src,f)
    name=f.split('.')[0].split('_')[0]
    directory_name = os.path.join(dst,name)
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    dst_file=os.path.join(directory_name,f)
    copyfile(src_file, dst_file)

    

        
        
