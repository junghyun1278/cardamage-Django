from django.conf import settings

from pycocotools.coco import COCO
import numpy as np
import matplotlib
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import random
pylab.rcParams['figure.figsize'] = (8.0, 10.0)# Import Libraries

# For visualization
import seaborn as sns
from matplotlib import colors
from tensorboard.backend.event_processing import event_accumulator as ea
from PIL import Image

# Scipy for calculating distance
from scipy.spatial import distance

import torch, torchvision

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
import skimage.io as io
from pathlib import Path

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device 객체

plt.rcParams["figure.figsize"] = [16,9]
register_coco_instances("car_data", {}, os.path.join(settings.BASE_DIR,'classification/inference/annotations.json'), os.path.join(settings.BASE_DIR,'classification/inference/'))

def segmenty(media_root, file_name):
    image_path = Path(media_root+'/'+file_name)
    #get configuration
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # only has one class (damage) + 1
    cfg.MODEL.RETINANET.NUM_CLASSES = 4 # only has one class (damage) + 1
    #cfg.MODEL.WEIGHTS = os.path.join("/home/adminuser/notebooks/01seokhwa/models/model_final.pth")
    cfg.MODEL.WEIGHTS = os.path.join(settings.BASE_DIR,'classification/inference/model/damage_final.pth')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 
    cfg['MODEL']['DEVICE']='cuda'#or cpu
    predictor = DefaultPredictor(cfg)

    damage_class_map= {0:'dent', 1:'scratch'}

    dataset = DatasetCatalog.get("car_data")
    metadata = MetadataCatalog.get("car_data")


    fig, (ax1) = plt.subplots(1, figsize =(16,12))

    img = io.imread(image_path)
    model_output = predictor(img)
    model_output = model_output["instances"].to("cpu")
    # only include detected instances with high enough score
    ni = model_output[model_output.pred_classes != 3]
    ni = ni[ni.pred_classes != 2]
    ni = ni[ni.scores > 0.8]

    # use the built-in visualizer to draw the boxes onto the image
    # v = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get("car_data"))
    v = Visualizer(img[:, :, ::], metadata=MetadataCatalog.get("car_data"))
    img = v.draw_instance_predictions(ni).get_image()


    try :
        os.remove(settings.SAVE_ROOT+'/detect.jpg')
    except :
        pass
    
    io.imsave(settings.SAVE_ROOT+'/'+ 'detect.jpg', img)

    return 'segmen'