Our MaskMitosis model is based on Mask RCNN

We use MaskMitosis12 model to segment the mitosis ground truth of the 2014 MITOSIS training set

1. Normalization of the 2014 MITOSIS training images before feeding them to MaskMitosis12 model for segmentation. We used the normalization toolbox proposed in https://github.com/mitkovetta/staining-normalization 
matlab normalization.m

2. Segmentation of the 2014 MITOSIS training set ground truth using MaskMitosis12 model, the list of the segmented candidates is saved as 'output/segmentation.txt' and the masks of the segmented candidates are saved as np arrays in 'output/masks' folder.
python ./mitosis.py detect --dataset=/path/to/dataset --subset=train_norm --weights=/path/to/MaskMitosis12/weights.h5
The normalized training set 'train_norm' should have the following structure:
    train_norm __ __ __ imageName __ __ __ images __ __ __ imageName.tiff

3. Saving the names of the mitosis ground truths as 'image_name_coordinates_bbox' in 'segmentation_train.txt'.
python ./ground_truth_sorting.py

4. generation of the masks of the full training images
python ./labels_generation.py








 

