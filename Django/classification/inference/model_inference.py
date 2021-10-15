import torch as torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import os
from PIL import Image
from pathlib import Path
from django.conf import settings as settings

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device 객체

import pandas as pd



def classify(media_root, file_name):
    car_dir = os.path.join(settings.BASE_DIR,'classification/inference')
    categories = pd.read_csv(car_dir+'/categories.csv', header = None)
    nb_classes = len(categories)
    categories = categories[0]
    categories = np.array(categories)

    image_path = Path(media_root+'/'+file_name)
    image = Image.open(image_path)
    transforms_test = transforms.Compose([transforms.Resize((224, 224)),                                   
                                    transforms.ToTensor(),                     
                                    transforms.Normalize(                      
                                    mean=[0.485, 0.456, 0.406],                
                                    std=[0.229, 0.224, 0.225])])
    
    test_datasets = transforms_test(image)
    test_datasets = torch.unsqueeze(test_datasets, 0)
    test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=1, shuffle=True, num_workers=4)

    model_dir = os.path.join(settings.BASE_DIR,'classification/inference/model/Best_model_car_efficient_b4_ver02_25.ph')
    model = torch.load(model_dir,map_location=device)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, nb_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model.eval()

    # Disable grad
    with torch.no_grad():
        running_loss = 0.
        running_corrects = 0
        for inputs in test_dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            classes = categories[preds[0]]
            
            car_model = [categories[preds[0]]]
            car_model = pd.DataFrame(car_model)

            try :
                os.remove(car_dir + '/car_model.csv')
            except :
                pass
            
            car_model.to_csv(car_dir + '/car_model.csv', index=False)

            return classes