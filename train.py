import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import gc
import argparse

from buildings import BuildingDataset, BuildingsConfig
from glob import glob
from os.path import join
from mrcnn import utils
import mrcnn.model as modellib

config = BuildingsConfig()

# Training dataset
dataset_train = BuildingDataset()
dataset_train.load_buildings(config.STEPS_PER_EPOCH, config.TRAIN_DIR)
dataset_train.prepare()

# Validation dataset
dataset_val = BuildingDataset()
dataset_val.load_buildings(config.VALIDATION_STEPS, config.VAL_DIR)
dataset_val.prepare()

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
                epochs=step[1], layers=step[0])

gc.collect() # prevent error upon system exit