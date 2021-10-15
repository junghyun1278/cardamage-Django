from django.conf import settings
from pathlib import Path


import torch
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
import torchvision
from torchvision import datasets, transforms, utils
import torchvision.models as models
from tqdm import tqdm
from PIL import Image

from sklearn import decomposition
from sklearn import manifold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
from torch.optim import lr_scheduler


from pycocotools.coco import COCO
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)# Import Libraries

# For visualization
import seaborn as sns
from matplotlib import colors
from tensorboard.backend.event_processing import event_accumulator as ea

# Scipy for calculating distance
from scipy.spatial import distance

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import os, json, cv2, random, time, copy

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
# Set base params
plt.rcParams["figure.figsize"] = [16,9]
register_coco_instances("cars_data", {}, os.path.join(settings.BASE_DIR,'classification/inference/annotations.json'), os.path.join(settings.BASE_DIR,'classification/inference/'))


def damaged_car(media_root, file_name):
    categories = ['damaged', 'no_damaged']
    
    image_path = Path(media_root+'/'+file_name)
    image = Image.open(image_path)
    
    transforms_test = transforms.Compose([transforms.Resize((224,224)),
                                         transforms.ToTensor(),
                                         torchvision.transforms.Grayscale(num_output_channels=3),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                        ])

    test_datasets = transforms_test(image)
    test_datasets = torch.unsqueeze(test_datasets, 0)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size = 1,shuffle = False, num_workers = 2)
    
    resnet = models.resnet50(pretrained = True)

    class Resnet(nn.Module):
        def __init__(self):
            super(Resnet, self).__init__()
            self.layer0 = nn.Sequential(*list(resnet.children())[0:-3])
            self.layer1 = nn.Sequential(*list(resnet.children())[-3:-1])
            self.layer2 = nn.Sequential(
            nn.Linear(2048,2)
            )

        def forward(self, x):
            out = self.layer0(x)
            out = self.layer1(out)
            out = out.view(1,-1)  # batch_size, -1
            out = self.layer2(out)
            return out

    model = Resnet().to(device)

    model_dir = os.path.join(settings.BASE_DIR,'classification/inference/model/Determination_damage.pth')
    model.load_state_dict(torch.load(model_dir))

    dataiter = iter(testloader)
    images = dataiter.next()

    images = images.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    
    print('Predicted: ', categories[predicted[0]])
    
    if predicted[0] != 0:
        io.imsave(settings.DAMAGE_ROOT+'/'+ 'car.jpg', image)
        damage_pred = 'non_damage'
        return damage_pred

    else : 
        #get configuration
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # only has one class (damage) + 1
        cfg.MODEL.RETINANET.NUM_CLASSES = 3 # only has one class (damage) + 1
        cfg.MODEL.WEIGHTS = os.path.join(settings.BASE_DIR,'classification/inference/model/car_damage_mask_rcnn.pth')
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 
        cfg['MODEL']['DEVICE']='cuda'#or cpu
        damage_predictor = DefaultPredictor(cfg)
        damage_class_map= {0:'dent', 1:'scratch'}

        dataset = DatasetCatalog.get("cars_data")
        metadata = MetadataCatalog.get("cars_data")

        fig, (ax1) = plt.subplots(1, figsize =(16,12))
        im = io.imread(image_path)

        #damage inference
        damage_outputs = damage_predictor(im)
        damage_v = Visualizer(im[:, :, ::-1],
                        metadata=metadata, 
                        scale = 1.3,                  
                        instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )

        damage_out = damage_v.draw_instance_predictions(damage_outputs["instances"].to("cpu"))

        damage_prediction_classes = [ damage_class_map[el] + "_" + str(indx) for indx,el in enumerate(damage_outputs["instances"].pred_classes.tolist())]
        damage_polygon_centers = damage_outputs["instances"].pred_boxes.get_centers().tolist()
        damage_dict = dict(zip(damage_prediction_classes,damage_polygon_centers))

        try :
            os.remove(settings.DAMAGE_ROOT+'/car.jpg')
        except :
            pass

        io.imsave(settings.DAMAGE_ROOT+'/'+ 'car.jpg', damage_out.get_image()[:, :, ::-1])
        segmen = list(damage_dict.keys())

        return segmen