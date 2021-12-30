# MaskMitosis
Codes for our Medical & Biological Engineering & Computing paper "MaskMitosis: a deep learning framework for fully supervised, weakly supervised, and unsupervised mitosis detection in histopathology images". Please see the [paper](https://link.springer.com/article/10.1007/s11517-020-02175-z) for more details.

# System Overview 
![GitHub Logo](/images/GA.jpg)

# Results
| Method                            | Evaluation set                   | *F*-score |
| ----------------------------------|:--------------------------------:| --------: |
| MaskMitosis12                     | 2012 ICPR MITOSIS test set       | 0.863     |
| MaskMitosis14 (weakly supervised) | 2014 ICPR MITOSIS validation set | 0.608     | 
| MaskMitosis14 (weakly supervised) | 2014 ICPR MITOSIS test set       | 0.475     | 
| MaskMitosis14 (unsupervised)      | 2014 ICPR MITOSIS validation set | 0.504     | 
| MaskMitosis14 (unsupervised)      | 2014 ICPR MITOSIS test set       | 0.395     | 

# Citing MaskMitosis
If you find MaskMitosis useful in your research, please consider citing:
```
@article{sebai2020maskmitosis,
title={MaskMitosis: a deep learning framework for fully supervised, weakly supervised, and unsupervised mitosis detection in histopathology images},
author={Sebai, Meriem and Wang, Xinggang and Wang, Tianjiang},
journal={Medical \& Biological Engineering \& Computing},
year={2020},
publisher={Springer}
}
```
# Content
1. [Hardware Requirements](#hardware-requirements)
2. [Software Requirements](#software-requirements)
3. [Installation](#installation)

# Hardware Requirements 
In our experiments, we used a Nvidia Quadro P5000 GPU with 16 GB of memory.

# Software Requirements 
Python 3.4, TensorFlow 1.3, Keras 2.0.8 and other common packages are required.

# Installation
Our mitosis detection and instance segmentation model is based on Mask RCNN model. We use the Matterport Inc implementation of Mask-RCNN. For more details see [https://github.com/matterport/Mask_RCNN]
1. Clone the repository.
2. Install dependencies: pip3 install -r requirements.txt
3. Run setup from the repository root directory: python3 setup.py install

The trained MaskMitosis12 and MaskMitosis14 models are available in [MaskMitosis12 model](https://drive.google.com/file/d/1hsb-SoTd4hImXu921F67XolJQD_6t13C/view?usp=sharing) and [MaskMitosis14 model](https://drive.google.com/file/d/1eGi8LM3Vgp_19wwcPOB1cS0YMNbSlKEn/view?usp=sharing), respectively.
