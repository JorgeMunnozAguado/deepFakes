
from csv_create import create_csv
from BinaryClassifier import BinaryClassifier
from Task1Loader import Task1_loader
from runner import *

from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.models as models

import os

def main():

    root_dir_train   = "./Task_1/"
    root_dir_test    = "./Task_2_3/"
    train_path       = os.path.join(root_dir_train, "development/")    # train images
    test_path        = os.path.join(root_dir_test, "evaluation/")    # train images

    create_csv(root_dir_train, root_dir_test, train_path, test_path)
    # preprocess()

    batch_size = 35

    model = models.resnet50(pretrained=True)
    clssf = BinaryClassifier(model)

    train_data = Task1_loader(root_dir_train + "train.csv", phase='train')
    test_data = Task1_loader(root_dir_test + "test.csv", phase='test')

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8)


    #criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.2)
    # optimizer = optim.SGD(model.parameters(), lr=0.0015, momentum=0.26)

    train(clssf, train_loader, valid_loader, criterion, optimizer, 10, device='cpu')


if __name__ == '__main__':
    main()
