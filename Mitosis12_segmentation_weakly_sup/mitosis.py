"""
Usage: 

    # To apply MaskMitosis12 to segment 2014 MITOSIS training set
    python ./mitosis.py detect --dataset=/path/to/dataset --subset=train_norm --weights=/path/to/MaskMitosis12/weights.h5

    The normalized training set 'train_norm' should have the following structure:
    train_norm __ __ __ imageName __ __ __ images __ __ __ imageName.tiff
"""

# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
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
import warnings
import scipy.misc

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
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "Mitosis12_segmentation_weakly_sup/logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "Mitosis12_segmentation_weakly_sup")

############################################################
#  Configurations
############################################################

class MitosisConfig(Config):
       
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
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1    
    IMAGE_RESIZE_MODE = "pad64"    
    IMAGE_MIN_SCALE = 2
    DETECTION_MIN_CONFIDENCE = 0.0 
    RPN_NMS_THRESHOLD = 0.7   


############################################################
#  Dataset
############################################################

class MitosisDataset(utils.Dataset):

    def load_mitosis(self, dataset_dir, subset):
        """Load a subset of the mitosis dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load. 
                * train_norm: images of the 2014 MITOSIS normalized training set                
        """
        # Add classes. We have one class.
        # Naming the dataset nucleus, and the class nucleus
        self.add_class("mitosis", 1, "mitosis")
        
        assert subset in ["train_norm"]
        subset_dir = subset
        dataset_dir = os.path.join(dataset_dir, subset_dir)
	
        image_ids = next(os.walk(dataset_dir))[1]
                    
        # Add images
        for image_id in image_ids:
            self.add_image(
            	"mitosis",
               	image_id=image_id,
                path=os.path.join(dataset_dir, image_id, "images/{}.tiff".format(image_id)))     
                

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

    masks_dir = "masks"
    masks_dir = os.path.join(submit_dir, masks_dir)
    if not os.path.exists(masks_dir):
        os.makedirs(masks_dir)		

    # Read dataset
    dataset = MitosisDataset()
    dataset.load_mitosis(dataset_dir, subset)
    dataset.prepare()
    
    submission = []
    file_path = os.path.join(submit_dir, "segmentation.txt")
    f=open(file_path, "w") 

    for image_id in dataset.image_ids:        
        image = dataset.load_image(image_id)
        # Detect mitoses
        r = model.detect([image], verbose=0)[0]        
        source_id = dataset.image_info[image_id]["id"]
        scores=np.array(r["scores"])
        bb=np.array(r['rois'])
        masks=np.array(r['masks'])
        for i in range(scores.shape[0]):  
            source_file=os.path.join('mitosis14_train','{0}'.format(source_id.split('_')[0]),source_id)
            f.write("{0} {1} {2} {3} {4} {5}\n".format(source_file,scores[i],bb[i][1],bb[i][0],bb[i][3],bb[i][2]))
            # saving each detected mask as np array
            mask_path = os.path.join(masks_dir, "{0}_{1}_{2}_{3}_{4}.txt".format(source_id,bb[i][1],bb[i][0],bb[i][3],bb[i][2]))
            fid=open(mask_path, "w")            
            np.savetxt(fid, masks[bb[i][0]:bb[i][2],bb[i][1]:bb[i][3],i], fmt='%d', delimiter=' ', newline='\n')
            fid.close()       
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
                        help="'detect'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Root directory of the 2014 MITOSIS dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file of the MaskMitosis12 model")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset', required=False,
                        metavar="Dataset sub-directory",
                        help="2014 MITOSIS training set : full images")
    args = parser.parse_args()

    # Validate arguments    
    if args.command == "detect":
        assert args.subset, "Provide --subset to run prediction on"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    # Configurations
    config = MitosisInferenceConfig()
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)

    # Select weights file to load
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
