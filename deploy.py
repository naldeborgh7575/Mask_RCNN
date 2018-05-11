"""Deploy model on target data. Usage: python deploy.py model target_data_dir"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import os
import numpy as np
import cv2
import gc

from buildings import BuildingsConfig, BuildingDataset
from tqdm import tqdm
from glob import glob
from os.path import join
from mrcnn import visualize
import mrcnn.model as modellib

def plt_cat(img, pred, save_to):
    """plot original image next to prediction (array)
    """
    base = '/'.join(save_to.split('/')[:-1])
    plt.figure(1)
    plt.subplot(121)
    plt.imshow(img)
    plt.title('Orig Image')
    plt.subplot(122)
    plt.imshow(pred.astype('uint8'))
    plt.title('Prediction')

    if not os.path.isdir(base):
        os.makedirs(base)
    plt.savefig(save_to)
    plt.close('all')
    return 1

config = BuildingsConfig()
weights = sys.argv[1]
base = join('/'.join(weights.split('/')[:-2]), 'deploy-results')
dataset = BuildingDataset()
dataset.load_buildings(sys.argv[2], 256) #count (500) is ignored
dataset.prepare()
print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

# Create output directory
if not os.path.exists(base): os.makedirs(base)

# Load model and weights
model = modellib.MaskRCNN(mode="inference", config=config, model_dir=config.MODEL_DIR)
model.load_weights(weights, by_name=True)

# Deploy in batches
for id_ix in tqdm(range(0,len(dataset.image_ids),config.BATCH_SIZE)):
    ids = [dataset.image_ids[i] for i in range(id_ix, id_ix + config.BATCH_SIZE)]
    imgs = [modellib.load_image_gt(dataset, config, i)[0] for i in ids]
    results = model.detect(imgs, verbose=0)

    for rix, r in enumerate(results):
        img_name = join(base, dataset.image_reference(ids[rix]))
        mi = visualize.display_instances(imgs[rix], r['rois'], r['masks'], r['class_ids'], 
                                         dataset.class_names, r['scores'], return_masked_image=True)
        plt_cat(imgs[rix], mi, img_name)


gc.collect() # prevent error upon system exit
