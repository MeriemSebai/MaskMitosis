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
        name_file=name
        obj_struct['imagename'] = [name_file]
        obj_struct['det'] = False
        obj_struct['det_name']=''
        objects.append(obj_struct)    
        line = f.readline()
        
    f.close()
    imagesetfile = 'train.txt'
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    class_recs = {}
    npos = 0
    for name in imagenames:
        imagename=name
        npos=npos+1
        class_recs[imagename]= [obj for obj in objects if obj['imagename'][0]==imagename]
        print (npos)
        
    return class_recs     
    
    
if __name__=='__main__':
    dithresh = 32
    width = 48  
 
    # read the gt
    test_anno_file = 'anno_train_14.txt'
    class_recs = parse_anno(test_anno_file)     
            
    # read segmentation    
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
        
    
    nd = len(image_ids)   #number of detections
    tp = np.zeros(nd)
    fp = np.zeros(nd)    
    
    for d in range(nd):   #for every detected bbs 
        if image_ids[d] not in class_recs:
            print (image_ids[d])
        else: 
            R = class_recs[image_ids[d]]  
            print (image_ids[d])
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
                    R[jmin]['det_name'] = '{}_{}_{}_{}_{}.txt'.format(image_ids[d].split('/')[2],int(bb[0]),int(bb[1]),int(bb[2]),int(bb[3]))   # Save the name of np array of the mask segmentation
                else:
                    fp[d] = 1.
            else:
                fp[d] = 1.

    pos=0 # number of segmented mitosis ground truths
    neg=0 # number of non segmented mitosis ground truths
    save_patch_dir='train_patches' # folder to save the patches that will be segmented with active contours
    if not os.path.exists(save_patch_dir):
        os.makedirs(save_patch_dir)
    for key in class_recs.keys():        
        for obj in class_recs[key]:
            print ('{} {} {} {}'.format(obj['centroid'],obj['imagename'],obj['det'],obj['det_name']))
            if obj['det']==1:
                pos=pos+1
            else: # if ground truth not segmented, save a patch around the centroid to segment it later with active contours
                neg=neg+1
                # crop the patches
                img=Image.open('/path/to/dataset/mitosis2014/mitosis14_train_norm/{}/{}.tiff'.format(obj['imagename'][0].split('/')[1],obj['imagename'][0].split('/')[2]))
                i=np.array(img)
                bb_x = int(obj['centroid'][0])
                bb_y = int(obj['centroid'][1])
                x1 = max(0, bb_x - width)
                if bb_x - width<0:
                    d1=-(bb_x - width)
                else:
                    d1=0
                x2 = min(1538, bb_x + width)
                if bb_x + width>1538:
                    d2=bb_x + width-1538
                else:
                    d2=0
                y1 = max(0, bb_y - width)
                if bb_y - width<0:
                    d3=-(bb_y - width)
                else:
                    d3=0
                y2 = min(1375, bb_y + width)
                if bb_y + width>1375:
                    d4=bb_y + width-1375
                else:
                    d4=0 
                obj['det_name']='{}_{}_{}_{}_{}.txt'.format(obj['imagename'][0].split('/')[2],x1,y1,x2,y2) # Save the name of np array of the mask segmentation as 'image_name_coordinates_bbox'
                obj['det']=1
                # save the patches in a folder
                img_crop = img.crop((x1,y1,x2,y2))
                patch = os.path.join(save_patch_dir,'{}_{}_{}_{}_{}_{}_{}_{}_{}.jpg'.format(obj['imagename'][0].split('/')[2],d1,d2,d3,d4,x1,y1,x2,y2))
                img_crop.save(patch)
    print ('pos {}'.format(pos))
    print ('neg {}'.format(neg)) 
    f=open('segmentation_train.txt','w') # file to save the name of the segmented mitosis ground truths 
    n=['']
    for key in class_recs.keys():
        for obj in class_recs[key]:                       
            if obj['det_name'] not in n:
                f.write('{}\n'.format(obj['det_name']))
    f.close()
    

            


