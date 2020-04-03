import os
import sys
import datetime
import numpy as np
import json
import skimage.draw


ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)
from pdfd.mrcnn.config import Config
from pdfd.mrcnn import model as modellib, utils

# path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argumentt --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

## Configurations
class CelebAConfig(Config):
    """
    Configuration for training on the CelbA dataset with Mask R-CNN.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "celeba"

    # Images per gpu
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 # Background + face

    # Number  of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9



## Dataseet
class CelebADataset(utils.Dataset):

    def load_ceba(self, dataset_dir, subset):
        """
        Load a subset of the CelebA dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes.
        self.add_class("celeba", 1, "face")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        
        # Load annotations.
        # Example form:
        # { 'filename': '39378403.jpg',
        #   'face': {
        #       'x': [...],
        #       'y': [...]
        #   }
        # }
        annotations = json.load(open(os.path.join(dataset_dir, "region_data.json")))
        annotations = list(annotations.values())

        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            if type(a['regions']) is dict:
                polygons = [r['']]

