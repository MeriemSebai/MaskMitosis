#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transforming the segmented mitosis ground truth images to np arrays and saving them in 'output/masks' folder.
"""
from PIL import Image
import numpy as np
import os
import matplotlib.image
import imageio

root_dir='train_patches_segmented'
save_dir='output/masks'
for f in os.listdir(root_dir):
    img = matplotlib.image.imread(os.path.join(root_dir,f))  
    mask=np.array(img)
    mask[mask>0]=1
    mask_path = os.path.join(save_dir, "{0}_{1}_{2}_{3}_{4}_{5}.txt".format(f.split('.')[0].split('_')[0],f.split('.')[0].split('_')[1],f.split('.')[0].split('_')[6],f.split('.')[0].split('_')[7],f.split('.')[0].split('_')[8],f.split('.')[0].split('_')[9]))
    fid=open(mask_path, "w") 
    np.savetxt(fid, mask, fmt='%d', delimiter=' ', newline='\n')
    fid.close()
