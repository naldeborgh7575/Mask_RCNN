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
model = modellib.MaskRCNN(mode="training", config=config,
                        model_dir=config.MODEL_DIR)

if config.RESTORE_FROM:
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    model.load_weights(config.RESTORE_FROM, by_name=True,
                        exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])

if config.FINE_TUNE:
    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE, 
                epochs=config.NUM_EPOCHS, 
                layers='heads')

else:
    # Retrain all layers
    # Passing layers="all" trains all layers. You can also 
    # pass a regular expression to select which layers to
    # train by name pattern.
    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE / 10,
                epochs=config.NUM_EPOCHS, 
                layers="all")

gc.collect() # prevent error upon system exit
