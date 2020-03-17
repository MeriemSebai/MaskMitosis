Our MaskMitosis model is based on Mask RCNN

Test:
1. To test MaskMitosis on the MITOSIS 2012 dataset
python ./mitosis.py detect --dataset=/path/to/dataset --subset=test --weights=</path/to/last/weights.h5

The test set 'test' should have the following structure:
    test __ __ __ imageName __ __ __ images __ __ __ imageName.bmp

2. To evaluate the detection result of MaskMitosis12 model, the names of the 2012 MITOSIS test images are listed in 'test.txt' and the mitosis ground truths are listed in 'anno_test_2012.txt' 
python Measure_detection_2012.py

 

