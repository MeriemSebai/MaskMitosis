Our MaskMitosis model is based on Mask RCNN

We use MaskMitosis12 model to segment the mitosis ground truth of the 2014 MITOSIS training set

1. Normalization of the 2014 MITOSIS training images before feeding them to MaskMitosis12 model for segmentation. We used the normalization toolbox proposed in https://github.com/mitkovetta/staining-normalization  
matlab normalization.m

2. Segmentation of the 2014 MITOSIS training set ground truth using MaskMitosis12 model, the list of the segmented candidates is saved as 'output/segmentation.txt' and the masks of the segmented candidates are saved as np arrays in 'output/masks' folder.
python ./mitosis.py detect --dataset=/path/to/dataset --subset=train_norm --weights=/path/to/MaskMitosis12/weights.h5
The normalized training set 'train_norm' should have the following structure:
    train_norm __ __ __ imageName __ __ __ images __ __ __ imageName.tiff

3. Keeping the segmented mitoses that correspond to the centroid ground truths and filtering out the remaining segmented candidates, the names of the mitosis ground truths are saved as 'image_name_coordinates_bbox' in 'segmentation_train.txt', the rest of the non segmented mitosis ground truths are saved as patches in 'train_patches' folder.
python ./ground_truth_sorting.py

4. segmentation of the remaining mitosis ground truths with active contours method.
matlab active_contour.m

5. transforming the segmented mitosis ground truth images to np arrays and saving them in 'output/masks' folder.
python imagetonp.py

5. generation of the masks of the full training images
python ./labels_generation.py








 

