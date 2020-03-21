import torch
import torch.nn as nn
import torch.optim as optim
import visdom
import numpy as np
import os
import models.LinearRegression as myLinear
import preprocess_Titanic as pre_Titanic


total_epoch = 500
criterion1 = nn.CrossEntropyLoss() #损失函数
use_cuda = torch.cuda.is_available()

net = myLinear.LinearRegression()

optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.1, momentum=0.5, weight_decay=0)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
train_set = []
train_label = []
test_set = []
test_label = []

best_Test_acc = 0.0
best_Test_acc_epoch = 0

model_path = 'train_models'



if __name__ == '__main__':
    if use_cuda:
        net.cuda()
    min_epoch = 1
    net = myLinear.LinearRegression()
    path = os.path.join(model_path, 'Test_model.t7')
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    # net.load_state_dict(checkpoint['net'], False)
    print('TestAcc:', checkpoint['best_Test_acc'])
    print('Epoch:', checkpoint['best_Test_acc_epoch'])
