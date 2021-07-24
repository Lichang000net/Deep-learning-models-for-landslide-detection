import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
import copy
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import time
# from tensorboardX import SummaryWriter
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import precision_score
# from sklearn.metrics import recall_score
# from sklearn.metrics import f1_score
# from torchvision import transforms
# from sklearn.metrics import precision_recall_curve


data_dir = "DataSet"

learning_rate = 0.0001
batch_size = 32
input_size = 224
epoch = 200
feature_extract = False
num_classes = 2


data_transform = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
        data_transform[x]) for x in ['train', 'val']}

dataloader_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
        batch_size=batch_size, shuffle=True, num_workers=0) for x in ["train", "val"]}

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda")


def set_param_requires_grad(model, feature_extract):
    count = 0
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False
            # print("fine tuning")
    elif not feature_extract:
        for param in model.parameters():
            param.requires_grad = True
            count += 1


def initialize_model(num_classes, feature_extract, use_pretrained=True):  #
        # resnet
        # model_ft = models.resnet50(pretrained=use_pretrained)
        # set_param_requires_grad(model_ft, feature_extract)  
        # num_fc_new = model_ft.fc.in_features
        # model_ft.fc = nn.Linear(num_fc_new, num_classes)
        # vgg
        model_ft = models.vgg16(pretrained=use_pretrained)
        set_param_requires_grad(model_ft, feature_extract)
        num_fc_new = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_fc_new, num_classes)
        # densenet
        # model_ft = models.densenet201(pretrained=use_pretrained)
        # set_param_requires_grad(model_ft, feature_extract)
        # num_fc_new = model_ft.classifier.in_features
        # model_ft.classifier = nn.Linear(num_fc_new, num_classes)
        return model_ft # , input_size


def train_model(model, dataloaders, loss_fn, optimizer, num_epoch=epoch):
    best_model_weight = copy.deepcopy(model.state_dict())
    best_acc = 0.
    val_acc_history = []
    val_loss = []

    for itration in range(num_epoch):
        for phase in ["train", "val"]:
            running_loss = 0.
            running_corrects = 0.
            if phase == "train":
                model.train()
            else:
                model.eval()

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.autograd.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)

                preds = outputs.argmax(dim=1)

                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds.view(-1) == labels.view(-1)).item()  

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            print("epoch {} in {}, loss: {}, acc: {}".format(itration+1, phase, epoch_loss, epoch_acc))

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weight = copy.deepcopy(model.state_dict())
                np.savetxt("model_" + str(epoch) + "_accur.txt", best_acc)
                model.load_state_dict(best_model_weight)
                model_path = "model_" + str(epoch) + "_.pkl"
                torch.save(model, model_path)

            if phase == "val":
                val_loss.append(epoch_loss)
                val_acc_history.append(epoch_acc)
                # weights = copy.deepcopy(model.state_dict())
                # model.load_state_dict(weights)
                # torch.save(model, temp_path)
            
    model.load_state_dict(best_model_weight)

    return val_loss, val_acc_history, model


def main():
    # model, input_size = initialize_model(num_classes, feature_extract, use_pretrained=True)
    model = initialize_model(num_classes, feature_extract, use_pretrained=True)
    # print(model)
    model = model.to(device)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=learning_rate, momentum=0.99)
    loss_fn = nn.CrossEntropyLoss()
    error, accur, model = train_model(model, dataloader_dict, loss_fn, optimizer, num_epoch=epoch)

    return error, accur, model


if __name__ == '__main__':
    loss, accur, model = main()
    model_path = "model.pkl"
    torch.save(model, model_path)


