"""train mask rcnn"""

import os
import re
import cv2
import numpy as np

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
    NAME = 'buildings'

    TRAIN_DIR = '/home/ubuntu/data/Mask_RCNN/data/samaww_60k/train/'
    VAL_DIR = '/home/ubuntu/data/Mask_RCNN/data/samaww_60k/validation/'
    MODEL_DIR = '/home/ubuntu/data/Mask_RCNN/models/'
    RESTORE_FROM = '/home/ubuntu/data/Mask_RCNN/models/mask_rcnn_coco.h5'
    FINE_TUNE = False

    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    NUM_EPOCHS = 100
    NUM_CLASSES = 2 # includes bg
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    TRAIN_ROIS_PER_IMAGE = 120

    TRAIN_SIZE = len(glob(join(TRAIN_DIR, 'buildings/*.png'))) + \
                 len(glob(join(TRAIN_DIR, 'buildings/*.jpg')))
    VALIDATION_SIZE = len(glob(join(VAL_DIR, 'buildings/*.png')))

    STEPS_PER_EPOCH = TRAIN_SIZE // (IMAGES_PER_GPU * GPU_COUNT)
    VALIDATION_STEPS = VALIDATION_SIZE // (IMAGES_PER_GPU * GPU_COUNT)

    def record(self, logdir):
        """Save config params in logdir
        """
        fname = join(logdir, self.NAME + '.txt')
        with open(fname, 'w') as f: f.write("Configurations:\n")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                with open(fname, 'a') as f: 
                    f.write("{:30} {}\n".format(a, getattr(self, a)))


class BuildingDataset(utils.Dataset):
    """Generates the buildings dataset
    """

    def load_buildings(self, count, data_dir, side_dim=256):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        data_dir: location of data. should have two subdirectories
            'buildings' (with images) and 'masks' (with masks)
        side_dim: the size of the generated images.
        """
        # Add classes
        self.add_class("buildings", 1, "building")

        # Add images
        im_paths = [join(data_dir, 'buildings', i) for i in os.listdir(join(data_dir, 'buildings'))]
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
        image_info = self.image_info[image_id]
        mask = cv2.imread(re.sub("buildings/", "masks/",
                                 image_info["path"]))
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

