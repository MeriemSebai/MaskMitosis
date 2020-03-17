import os
import pickle
import _pickle as cPickle
import numpy as np
import math
from PIL import Image

def parse_anno(filename):
    objects = []
    
    f=open(filename,'r')
    line = f.readline()
    while line:
        line=line.split(' ')
        obj_struct = {}
        name=line[0]
        cen_x = line[1]
        cen_y = line[2]
        obj_struct['centroid']=[cen_x,cen_y]
        obj_struct['imagename'] = [name]
        obj_struct['det'] = False
        objects.append(obj_struct)    
        line = f.readline()
        
    f.close()
    imagesetfile = './validation.txt' # validation.txt contains the name of the 2014 MITOSIS validation images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        class_recs[imagename]= [obj for obj in objects if obj['imagename'][0]==imagename]
        
    return class_recs     
    
    
if __name__=='__main__':
    dithresh = 32 # detection threshold
     
    # read the ground truths
    test_anno_file = './anno_validation_2014.txt'
    class_recs = parse_anno(test_anno_file) 
            
    # read detections    
    with open('./output/detections.txt','r') as f:
        lines = f.readlines()
               
    splitlines = [x.strip().split() for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
            
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB=BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]
    sorted_scores = -sorted_scores  
  
    # Mark TPs and FPs
    nd = len(image_ids)   #number of detections
    tp = np.zeros(nd)
    fp = np.zeros(nd)    
    
    for d in range(nd):   #for every detected bbs        
        R = class_recs[image_ids[d]]          
        BBGT =[x['centroid'] for x in R]
        BBGT = np.transpose(np.array(BBGT).astype(float))
        
        distmin = np.inf
        bb = BB[d, :].astype(float)
        bb_x = (bb[0] + bb[2])/2
        bb_y = (bb[1] + bb[3])/2
                        
        if BBGT.size > 0:
            dist = np.sqrt(np.square(BBGT[0]-bb_x) +np.square(BBGT[1]-bb_y))
            distmin = np.min(dist)
            jmin = np.argmin(dist)
            
        if distmin < dithresh:
            if not R[jmin]['det']:
                tp[d] = 1.
                R[jmin]['det'] = 1               
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.
    
    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    npos = 163
    rec = tp / float(npos)
    prec = tp /np.maximum(tp + fp, np.finfo(np.float64).eps)
    F= 2*rec*prec/(rec+prec)
    F = [x for x in F if not math.isnan(x)]
    F_max=np.max(F)    
    max_index = F.index(F_max)
    score_thresh = sorted_scores[max_index]
    prec_m=prec[max_index]
    rec_m = rec[max_index]
    print ('the score threshold is {}'.format(score_thresh))
    print('the max F is {}'.format(F_max))   
    print('the prec is {}, and the rec is {}'.format(prec_m, rec_m))
    


            


