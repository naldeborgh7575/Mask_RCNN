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
from mrcnn.visualize import *
import mrcnn.model as modellib

def plot_instances(image, boxes, masks, class_ids, save_to):
    """plot original image next to prediction
    """
    base = '/'.join(save_to.split('/')[:-1])

    # Number of instances
    N = boxes.shape[0]
    assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
    colors = random_colors(N)

    # Create figure
    plt.figure(1)
    plt.subplot(121); plt.imshow(image); plt.title('Orig Image')
    ax = plt.subplot(122)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')

    masked_image = image.astype(np.uint32).copy()

    for i in range(N):
        color = colors[i]
        mask = masks[:, :, i]
        masked_image = apply_mask(masked_image, mask, color)

    ax.imshow(masked_image.astype(np.uint8))

    if not os.path.isdir(base): os.makedirs(base)
    plt.savefig(save_to); plt.close('all')
    return 1

config = BuildingsConfig()
weights = sys.argv[1]
base = join('/'.join(weights.split('/')[:-2]), 'deploy-results')
dataset = BuildingDataset()
dataset.load_buildings(sys.argv[2], 256) 
dataset.prepare()
print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

# Create output directory
if not os.path.exists(base): os.makedirs(base)

# Load model and weights
model = modellib.MaskRCNN(mode="inference", config=config, model_dir=config.MODEL_DIR)
model.load_weights(weights, by_name=True)
print('Loaded model weights ' + weights.split('/')[-1])

# Deploy in batches
for id_ix in tqdm(range(0,len(dataset.image_ids),config.BATCH_SIZE)):
    ids = [dataset.image_ids[i] for i in range(id_ix, id_ix + config.BATCH_SIZE)]
    imgs = [modellib.load_image_gt(dataset, config, i)[0] for i in ids]
    results = model.detect(imgs, verbose=0)

    for rix, r in enumerate(results):
        img_name = join(base, dataset.image_reference(ids[rix]))
        if len(sys.argv) > 3:
            if sys.argv[3] == 'p':
                plot_instances(imgs[rix], r['rois'], r['masks'], r['class_ids'], 
                            img_name[:-4] + '_plt.png')
        else: # Save instance mask
            ints = np.random.choice(np.arange(1,256), size=r['masks'].shape[-1],
                                    replace=False)
            mask = np.zeros((256,256))
            for inst_ix in range(r['masks'].shape[-1]):
                instance = r['masks'][:,:,inst_ix]
                mask[instance] = ints[inst_ix]
            cv2.imwrite(img_name, mask*3)


gc.collect() # prevent error upon system exit

