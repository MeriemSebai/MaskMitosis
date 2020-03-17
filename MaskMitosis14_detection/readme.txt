Train:
To train MaskMitosis on the 2014 MITOSIS dataset:
python ./mitosis.py train --dataset=/path/to/dataset --subset=train --weights=imagenet  
    
The training set 'train' should have the following structure:
train __ __ __ imagePatchName __ __ __ images __ __ __ imagePatchName.jpg
                     |
                     |__ __ __ __ __ __ __ masks __ __ __ imagePatchName_1.jpg
                                             |
                                             |__ __ __ __ imagePatchName_2.jpg 
                                             |
                                             |__ __ __ __ imagePatchName_n.jpg   

To test MaskMitosis on the 2014 MITOSIS dataset:
python ./mitosis.py detect --dataset=/path/to/dataset --subset=(test/validation) --weights=/path/to/last/weights.h5

The validation set 'validation' and the test set 'test' should have the following structure:
(validation/test) __ __ __ imageName __ __ __ images __ __ __ imageName.tiff

2. To evaluate the detection result of MaskMitosis14 model, the names of the 2014 MITOSIS validation images are listed in 'validation.txt' and the mitosis ground truths are listed in 'anno_validation_2014.txt' 
python Measure_detection_2014.py


 

