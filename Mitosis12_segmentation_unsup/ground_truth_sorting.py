import os
import pickle
import _pickle as cPickle
import numpy as np
import math
from PIL import Image

  
with open('output/segmentation.txt','r') as f:
    lines = f.readlines()
                
splitlines = [x.strip().split() for x in lines]
image_ids = [x[0] for x in splitlines]
confidence = np.array([float(x[1]) for x in splitlines])
BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
            
#sort by confidence
sorted_ind = np.argsort(-confidence)
sorted_scores = np.sort(-confidence)
BB=BB[sorted_ind, :]
image_ids = [image_ids[x] for x in sorted_ind]

f=open('segmentation_train.txt','w')   
nd = len(image_ids)   #number of detections
for d in range(nd):   #for every detected bbs 
    bb = BB[d, :].astype(float)
    obj = '{}_{}_{}_{}_{}.txt'.format(image_ids[d].split('/')[2],int(bb[0]),int(bb[1]),int(bb[2]),int(bb[3]))     
    f.write('{}\n'.format(obj))
f.close()
    

            

