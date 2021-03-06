
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
    root_dir_test    = "./Task_2_3/"
    train_path       = os.path.join(root_dir_train, "development/")    # train images
    test_path        = os.path.join(root_dir_test, "evaluation/")    # train images

    create_csv(root_dir_train, root_dir_test, train_path, test_path)

    preprocess = FaceRecog(margin=7)

    batch_size = 35

    model = models.resnet34(pretrained=False)
    clssf = BinaryClassifier(model, freeze=False)

    train_data = Task1_loader(root_dir_train + "train.csv", phase='train', preprocess=preprocess)
    test_data = Task1_loader(root_dir_test + "test.csv", phase='test', preprocess=preprocess)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8)


    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0018, weight_decay=0.0015)

    load_checkpoint(model, 'checkpoints/task1_84_7.pkl', optimizer)

    valid_loss, accuracy = validate(model, valid_loader, criterion, 'cpu')

    print(f'val loss: {valid_loss:04f}  '
          f'val acc: {valid_acc*100:.4f}%')


if __name__ == '__main__':
    main()
