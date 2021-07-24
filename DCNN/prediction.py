from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from torchvision import transforms
import torch
from PIL import Image
import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import random
import csv
import codecs
# from tqdm import tqdm_notebook
from tqdm.notebook import tqdm
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from torchvision import transforms

    

def getTruth():
    land = np.ones((landslide_num))
    ground = np.zeros((ground_num))
    res = np.hstack((land, ground))
    return res

truth = getTruth()

def OutputIndex(y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("tp:", tp, "fn:", fn, "tn:", tn, "fp:", fp)
    acc = (tp + tn) / samples

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print("Aaccuracy,{:.2f}  Precision,{:.2f}  Recall,{:.2f}  F1-Score,{:.2f}".format(acc, precision, recall, f1))
    increasedNum(tp, fn, tn, fp)

# torch.no_grad()
transform = transforms.Compose([
transforms.Resize(256),
# transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406],
std=[0.229, 0.224, 0.225])
])

Name = ["vgg16", "vgg19", "res50", "res101", "dense121", "dense201"] 
Amount = ["TotalS","S1000", "S100"]

val_path = "DCNN/dataset"

filenames = os.listdir(val_path)
device = torch.device("cuda")

Path = ["landslides", "ground"]


for num in Amount:
    for name in Name:
        res = []
        print("***********************************************")
        for path in Path:
            model_path = "model.pkl"
            net = torch.load(model_path, map_location='cuda:0')
            net = net.to(device)
            net.eval()
            print(num, " ", name)
            for n, id_ in tqdm(enumerate(filenames), total=len(filenames), disable=False):
                img_path = path + id_
                img = Image.open(img_path)
                img = transform(img).unsqueeze(0)
                img_ = img.to(device)
                outputs = net(img_)
                _, predicted = torch.max(outputs, 1)
                res.append(predicted[0].cpu().detach().numpy())
        matrix = confusion_matrix(truth, res)
        print(matrix)
        OutputIndex(truth, res)
        
        
        
   