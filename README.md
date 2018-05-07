# Mask R-CNN for Object Detection and Segmentation

Mask RCNN for building instance segmentation

# Environment

With the configuration in requrements.txt use NVIDIA driver 384.111, cudnn 6 and cuda-8.0

```bash
virtualenv --system-site-packages -p python3 ~/.venv/mrcnn
source ~/.venv/mrcnn/bin/activate
pip install -r requirements.txt
```

# Data prep

From semantic segmentation masks (0: no building, 1: building)
```bash
python to_instance.py /path/to/mask/dir/
```