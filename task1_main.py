
from csv_create import create_csv
from BinaryClassifier import BinaryClassifier
from Task1Loader import Task1_loader
from runner import *
from face_recognition import FaceRecog

from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.models as models

import os

def main():

    root_dir_train   = "./Task_1/"
    root_dir_test    = "./Task_1/"
    train_path       = os.path.join(root_dir_train, "development/")    # train images
    test_path        = os.path.join(root_dir_test, "evaluation/")    # train images

    create_csv(root_dir_train, root_dir_test, train_path, test_path)
    
    preprocess = FaceRecog(margin=7)
    # preprocess = None

    batch_size = 55

    # model = models.resnet18(pretrained=False)
    model = models.resnet34(pretrained=False)
    clssf = BinaryClassifier(model, freeze=False)

    train_data = Task1_loader("./Task_1/train.csv", phase='train', preprocess=preprocess)
    test_data = Task1_loader("./Task_1/test.csv", phase='test', preprocess=preprocess)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8)


    criterion = nn.BCELoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.0018, momentum=0.27)
    optimizer = optim.Adam(model.parameters(), lr=0.0018, weight_decay=0.0015)

    train(clssf, train_loader, valid_loader, criterion, optimizer, 10, device='cpu')


if __name__ == '__main__':
    main()
