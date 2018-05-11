import os
import sys
import random
import math
import re
import time
import imgaug
import numpy as np
import cv2
import gc
import argparse
import subprocess

from buildings import BuildingDataset, BuildingsConfig
from glob import glob
from os.path import join
from mrcnn import utils
import mrcnn.model as modellib

config = BuildingsConfig()

# Training dataset
dataset_train = BuildingDataset()
dataset_train.load_buildings(config.TRAIN_DIR)
dataset_train.prepare()

# Validation dataset
dataset_val = BuildingDataset()
dataset_val.load_buildings(config.VAL_DIR)
dataset_val.prepare()

# Define augmentation
if config.AUGMENTATION:
    augmentation = imgaug.augmenters.Sometimes(0.5, [
        imgaug.augmenters.Fliplr(0.5),
        imgaug.augmenters.Flipud(0.5),
        imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
    ])
else: augmentation = None

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config, model_dir=config.MODEL_DIR)

if config.RESTORE_FROM:
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    model.load_weights(config.RESTORE_FROM, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])

for step in config.TRAINING_SCHEDULE:
    assert(len(step) == 3)
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE/step[-1],
                epochs=step[1], layers=step[0], augmentation=augmentation)

gc.collect() # prevent error upon system exit

# upload to s3
command = 'aws s3 sync ~/Mask_RCNN/experiments/ s3://nja-data/Mask_RCNN/experiments/'
proc = subprocess.Popen([command], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = proc.communicate()

# Shutdown instance
if len(sys.argv) > 1:
    if sys.argv[1] == 'shutdown':
        subprocess.Popen(['sudo shutdown -h now'], shell=True)