"""train mask rcnn"""

import os
import re
import cv2
import numpy as np
import pandas as pd

from mrcnn.config import Config
from mrcnn import utils
from glob import glob
from os.path import join

class BuildingsConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = '53118augc10k'

    DATA_DIR = '/home/ubuntu/data/Mask_RCNN/data/sama_d5.csv'
    MODEL_DIR = join('/home/ubuntu/data/Mask_RCNN/experiments/', NAME)
    RESTORE_FROM = '/home/ubuntu/data/Mask_RCNN/pretrained_weights/crowdai.h5'
    FINE_TUNE = False
    OPTIMIZER = 'sgd' # sgd or adam
    AUGMENTATION = True

    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Training schedule: [[layers_n, epochs_n, lrfrac_n], ...] n = train step
    # Layer Options:
    #   all: All the layers
    #   3+: Train Resnet stage 3 and up
    #   4+: Train Resnet stage 4 and up
    #   5+: Train Resnet stage 5 and up
    # lrfrac = fraction of LEARNING_RATE to use
    TRAINING_SCHEDULE = [['heads',1,1],['4+',10,1],['all', 40, 10]]
    NUM_EPOCHS = sum([i[1] for i in TRAINING_SCHEDULE])

    NUM_CLASSES = 2 # includes bg
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimzer
    # implementation.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0001
    LR_DECAY = 0.0

    # Backbone: resnet 101 or resnet 50
    BACKBONE = "resnet101"

    # Loss weights for more precise optimization.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # square anchor side in pixels
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    TRAIN_ROIS_PER_IMAGE = 120

    train_set = pd.read_csv(DATA_DIR)
    TRAIN_SIZE = len(train_set[train_set['set'] == 'tr'])
    VALIDATION_SIZE = len(train_set[train_set['set'] == 'val'])

    STEPS_PER_EPOCH = TRAIN_SIZE // (IMAGES_PER_GPU * GPU_COUNT)
    VALIDATION_STEPS = VALIDATION_SIZE // (IMAGES_PER_GPU * GPU_COUNT)


class BuildingDataset(utils.Dataset):
    """Generates the buildings dataset
    """

    def load_buildings(self, data, side_dim=256, validation=False):
        """Generate the requested number of images.
        data: csv of file paths, requires mask, image, set 
            columns. set is val or tr. mask and image have full
            path.
        side_dim: the size of the generated images.
        """
        # Add classes
        self.add_class("buildings", 1, "building")

        # Add images
        self.tvdata = pd.read_csv(data)
        self.val = validation

        if validation: im_paths = list(self.tvdata['image'][self.tvdata['set']=='val'])
        else: im_paths = list(self.tvdata['image'][self.tvdata['set']=='tr'])

        for ix, path in enumerate(im_paths):
            self.add_image("buildings", image_id=ix, path=path,
                           width=side_dim, height=side_dim, bg_color=np.array([0,0,0]))

    def load_mask(self, image_id):
        """Generate a mask for the corresponding image id.
        Masks are stored as 3d arrays in a directory called 'masks'
            with each instance encoded as a unique shade of gray.
        image_id: image id assigned in load_buildings
        returns array [h, w, instances]
        """
        if self.val: im_paths = list(self.tvdata['mask'][self.tvdata['set']=='val'])
        else: im_paths = list(self.tvdata['mask'][self.tvdata['set']=='tr'])

        image_info = self.image_info[image_id]
        mask = cv2.imread(im_paths[image_id])
        instances = sorted(np.unique(mask))[1:]

        # Reformat mask
        new_mask = np.zeros([image_info['height'],
                             image_info['width'],
                             len(instances)])

        for ix, inst in enumerate(instances):
            new_mask[:, :, ix][mask[:, :, 0] == inst] = 1

        # All class names will be one
        return new_mask, np.ones(len(instances)).astype(np.int32)

    def image_reference(self, image_id):
        """Give +vivid image name for a given image id for debugging
        """
        return self.image_info[image_id]["path"].split('/')[-1]


