"""
Usage: 

    # To train MaskMitosis on the 2012 MITOSIS dataset:    
    python ./mitosis.py train --dataset=/path/to/dataset --subset=train --weights=imagenet 

    The training set 'train' should have the following structure:
    train __ __ __ imagePatchName __ __ __ images __ __ __ imagePatchName.jpg
                     |
                     |__ __ __ __ __ __ __ masks __ __ __ imagePatchName_1.jpg
                                             |
                                             |__ __ __ __ imagePatchName_2.jpg 
                                             |
                                             |__ __ __ __ imagePatchName_n.jpg          

    # To test MaskMitosis on the MITOSIS 2012 dataset
    python ./mitosis.py detect --dataset=/path/to/dataset --subset=test --weights=/path/to/final/weights.h5

    The test set 'test' should have the following structure:
    test __ __ __ imageName __ __ __ images __ __ __ imageName.bmp
                     
"""

if __name__ == '__main__':
    import matplotlib
    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import os
import sys
import json
import datetime
import numpy as np
import skimage.io
from imgaug import augmenters as iaa
import warnings

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils as utils
from mrcnn import model as modellib
from mrcnn import visualize as visualize

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "MaskMitosis12_detection/logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "MaskMitosis12_detection")

############################################################
#  Configurations
############################################################

class MitosisConfig(Config):
    """Configuration for training on the mitosis dataset."""    
    NAME = "mitosis"
    IMAGES_PER_GPU = 2    
    NUM_CLASSES = 2    
    MINI_MASK_SHAPE = (224, 224)
    TRAIN_ROIS_PER_IMAGE = 512
    RPN_NMS_THRESHOLD = 0.9     
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024  	 
     

class MitosisInferenceConfig(MitosisConfig):
    """Configuration for inference on the mitosis dataset."""
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 4096
    IMAGE_MAX_DIM = 4096
    RPN_NMS_THRESHOLD = 0.7   


############################################################
#  Dataset
############################################################

class MitosisDataset(utils.Dataset):

    def load_mitosis(self, dataset_dir, subset):
        """Load a subset of the mitosis dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load. 
                * train: images of the training set 
                * test: images of the test set
        """
        # Add classes. We have one class.
        # Naming the dataset mitosis, and the class mitosis
        self.add_class("mitosis", 1, "mitosis")

        assert subset in ["train", "test"]
        subset_dir = subset
        dataset_dir = os.path.join(dataset_dir, subset_dir)	
        image_ids = next(os.walk(dataset_dir))[1]
                    
        # Add images
        for image_id in image_ids:
            if subset_dir=="train":
                self.add_image(
                        "mitosis",
                        image_id=image_id,
                        path=os.path.join(dataset_dir, image_id, "images/{}.jpg".format(image_id)))
            else:
                self.add_image(
                        "mitosis",
                        image_id=image_id,
                        path=os.path.join(dataset_dir, image_id, "images/{}.bmp".format(image_id)))
                

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        # Get mask directory from image path
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")

        # Read mask files from .jpg image
        mask = []
        for f in next(os.walk(mask_dir))[2]:
            if f.endswith(".jpg"):
                m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
                mask.append(m)
        mask = np.stack(mask, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "mitosis":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################

def train(model, dataset_dir, subset):
    """Train the model."""
    # Training dataset.
    dataset_train = MitosisDataset()
    dataset_train.load_mitosis(dataset_dir, subset)
    dataset_train.prepare()    

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.Fliplr(0.5)

    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    # Training - Stage 1
    print("Training network heads")
    model.train(dataset_train,
                learning_rate=config.LEARNING_RATE,
                epochs=50,
                layers='heads',
                augmentation=augmentation)

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train,
                learning_rate=config.LEARNING_RATE,
                epochs=150,
                layers='4+',
                augmentation=augmentation)

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=200,
                layers='all',
                augmentation=augmentation)


############################################################
#  Detection
############################################################

def detect(model, dataset_dir, subset):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    submit_dir = "output"
    submit_dir = os.path.join(RESULTS_DIR, submit_dir)
    if not os.path.exists(submit_dir):
        os.makedirs(submit_dir)    

    # Read dataset
    dataset = MitosisDataset()
    dataset.load_mitosis(dataset_dir, subset)
    dataset.prepare()
    # Load over images
    submission = []
    file_path = os.path.join(submit_dir, "detections.txt")
    f=open(file_path, "w") 
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect mitoses
        r = model.detect([image], verbose=0)[0]
        # save the detections
        source_id = dataset.image_info[image_id]["id"]
        scores=np.array(r["scores"])
        bb=np.array(r['rois'])
        for i in range(scores.shape[0]):  
            source_file=os.path.join('mitosis12_test','{0}_v2'.format(source_id.split('_')[0]),source_id)
            f.write("{0} {1} {2} {3} {4} {5}\n".format(source_file,scores[i],bb[i][1],bb[i][0],bb[i][3],bb[i][2]))        
    f.close()    

############################################################
#  Command Line
############################################################

if __name__ == '__main__':
    import argparse
    warnings.filterwarnings("ignore")

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for mitosis detection')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = MitosisConfig()
    else:
        config = MitosisInferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ...")
    model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset, args.subset)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
