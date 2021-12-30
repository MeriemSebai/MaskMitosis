Train:
To train MaskMitosis on the 2012 MITOSIS dataset:
python ./mitosis.py train --dataset=/path/to/dataset --subset=train --weights=imagenet 

The training set 'train' should have the following structure:
    train __ __ __ imagePatchName __ __ __ images __ __ __ imagePatchName.jpg
                     |
                     |__ __ __ __ __ __ __ masks __ __ __ imagePatchName_1.jpg
                                             |
                                             |__ __ __ __ imagePatchName_2.jpg 
                                             |
                                             |__ __ __ __ imagePatchName_n.jpg

Test:
1. To test MaskMitosis on the 2012 MITOSIS dataset
python ./mitosis.py detect --dataset=/path/to/dataset --subset=test --weights=/path/to/last/mitosis2012_weights.h5

The test set 'test' should have the following structure:
    test __ __ __ imageName __ __ __ images __ __ __ imageName.bmp

2. To evaluate the detection result of MaskMitosis model, the names of the 2012 MITOSIS test images are listed in 'test.txt' and the mitosis ground truths are listed in 'anno_test_2012.txt' 
python Measure_detection_2012.py

 

