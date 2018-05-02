import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import argparse

from config import Config
from building_dataset import BuildingDataset
from os.path import join
import utils
import model as modellib

MODEL_DIR = join(os.getcwd(), "models/122917_ptcoco/")
MODEL_NAME = "delivery_1"
RESTORE_FROM = join(os.getcwd(), "mask_rcnn_coco.h5")
TRAIN_SIZE = 67992
VAL_SIZE = 3000
NUM_EPOCHS = 1
IMGS_PER_GPU = 8
NUM_CLASSES = 2
INPUT_SIZE = '256,256'
NUM_GPU = 1
TRAIN_DIR = '/home/ubuntu/mrcnn_data/delivery_1/train/'
VAL_DIR = '/home/ubuntu/mrcnn_data/delivery_1/validation/'

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--model-dir", type=str, default=MODEL_DIR,
                        help="Path to the directory in which to save models.")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME,
                        help="Name of the model.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Path to the model weights to load.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--train-size", type=int, default=TRAIN_SIZE,
                        help="Number of training samples.")
    parser.add_argument("--val-size", type=int, default=VAL_SIZE,
                        help="Number of validation samples.")
    parser.add_argument("--num-epoch", type=int, default=NUM_EPOCHS,
                        help="Number of training epochs.")
    parser.add_argument("--imgs-per-gpu", type=int, default=IMGS_PER_GPU,
                        help="Number of images to load on one gpu during training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes (inluding background).")
    parser.add_argument("--num-gpu", type=int, default=NUM_GPU,
                        help="Number of GPUs to use for training.")
    parser.add_argument("--train-dir", type=str, default=TRAIN_DIR,
                        help="location of training data.")
    parser.add_argument("--val-dir", type=str, default=VAL_DIR,
                        help="location of validation data.")
    return parser.parse_args()

args = get_arguments()
args.input_size = [int(i) for i in args.input_size.split(',')]

# Model configurations
class BuildingsConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = args.model_name

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 15 (GPUs * images/GPU).
    GPU_COUNT = args.num_gpu
    IMAGES_PER_GPU = args.imgs_per_gpu

    # Number of classes (including background)
    NUM_CLASSES = args.num_classes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = args.input_size[0]
    IMAGE_MAX_DIM = args.input_size[0]

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    TRAIN_ROIS_PER_IMAGE = 120

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = args.train_size // (args.num_gpu * args.imgs_per_gpu)
    VALIDATION_STEPS = args.val_size // (args.num_gpu * args.imgs_per_gpu)


if __name__ == '__main__':
    config = BuildingsConfig()

    # Training dataset
    dataset_train = BuildingDataset()
    dataset_train.load_buildings(args.train_size, args.train_dir)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = BuildingDataset()
    dataset_val.load_buildings(args.val_size, args.val_dir)
    dataset_val.prepare()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                            model_dir=args.model_dir)

    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    model.load_weights(args.restore_from, by_name=True,
                    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE, 
                epochs=1, 
                layers='heads')
