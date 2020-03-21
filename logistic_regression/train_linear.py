import torch
import torch.nn as nn
import torch.optim as optim
import visdom
import numpy as np
import os
import models.LinearRegression as myLinear
import preprocess_Titanic as pre_Titanic


total_epoch = 1000
criterion1 = nn.CrossEntropyLoss()# 损失函数
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

vis = visdom.Visdom(env='my_Logistic_Regression_model')
vis.text('Epoch\tTrain_loss\tTrain_acc\tTest_loss\tTest_acc', win='train_info')
def my_train(epoch, min_epoch, train_set, train_label):
    if use_cuda:
        train_x = torch.from_numpy(train_set).float().cuda()
        train_y = torch.from_numpy(train_label).long().cuda()
    else:
        train_x = torch.from_numpy(train_set).float()
        train_y = torch.from_numpy(train_label).long()
    outputs = net(train_x)
    loss = criterion1(outputs, train_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step(loss)

    # 计算train_acc
    _, predicted = torch.max(outputs.data, 1)
    total = train_x.size(0)
    correct = predicted.eq(train_y.data).cpu().sum()

    if (epoch + 1) % min_epoch == 0:
        print("lr:%f"%(optimizer.param_groups[0]["lr"]))
        print("Epoch %d/%d, train_loss:%.3f, train_acc:%.3f%%(%d/%d)"%(epoch + 1,total_epoch, loss,
                                                                  correct * 100.0/ total, correct, total))
        Train_acc = correct * 100.0 / total
        if epoch + 1 == min_epoch:
            vis.line(X=np.array([(epoch + 1) / min_epoch]),
                     Y=np.array([Train_acc.item()]),
                     win='train_acc')
        else:
            vis.line(X=np.array([(epoch + 1) / min_epoch]),
                     Y=np.array([Train_acc.item()]),
                    win = 'train_acc', update = 'append')

        if epoch + 1 == 50:
            vis.line(X=np.array([(epoch + 1) / min_epoch]),
                     Y=np.array([loss.item()]),
                     win='train_loss')
        else:
            vis.line(X=np.array([(epoch + 1) / min_epoch]),
                     Y=np.array([loss.item()]),
                    win = 'train_loss', update = 'append')
        # if epoch + 1 == 50:
        #     vis.line(X=np.column_stack((np.array([(epoch + 1) / 50]), np.array([(epoch + 1) / 100]))),
        #              Y=np.column_stack(
        #                  (np.array([Train_acc.item()]), np.array([loss.item()]))),
        #              win='train_acc_loss')
        # else:
        #     vis.line(X=np.column_stack((np.array([(epoch + 1) / 50]), np.array([(epoch + 1) / 100]))),
        #              Y=np.column_stack(
        #                  (np.array([Train_acc.item()]), np.array([loss.item()]))),
        #              win='train_acc_loss', update='append')
    return loss.item(), correct.item() / total

def my_test(epoch, min_epoch, test_set, test_label):
    global best_Test_acc
    global best_Test_acc_epoch
    global model_path
    if use_cuda:
        test_x = torch.from_numpy(test_set).float().cuda()
        test_y = torch.from_numpy(test_label).long().cuda()
    else:
        test_x = torch.from_numpy(test_set).float()
        test_y = torch.from_numpy(test_label).long()
    outputs = net(test_x)
    loss = criterion1(outputs, test_y)

    _, predicted = torch.max(outputs.data, 1)
    total = test_x.size(0)
    correct = predicted.eq(test_y.data).cpu().sum()
    Test_acc = correct * 100.0 / total

    if Test_acc > best_Test_acc:
        print('Saving..')
        print("best_Test_acc: %0.3f" % Test_acc)
        state = {'net': net.state_dict() if use_cuda else net,
                 'best_Test_acc': Test_acc,
                 'best_Test_acc_epoch': epoch,
                 }
        if not os.path.isdir(model_path):
            os.mkdir(model_path)
        torch.save(state, os.path.join(model_path, 'Test_model.t7'))
        best_Test_acc = Test_acc
        best_Test_acc_epoch = epoch

    if epoch + 1 == min_epoch:
        vis.line(X=np.array([(epoch + 1) / min_epoch]),
                 Y=np.array([Test_acc.item()]),
                 win='test_acc')
    else:
        vis.line(X=np.array([(epoch + 1) / min_epoch]),
                 Y=np.array([Test_acc.item()]),
                 win='test_acc',update='append')
    if epoch + 1 == min_epoch:
        vis.line(X=np.array([(epoch + 1) / min_epoch]),
                 Y=np.array([loss.item()]),
                 win='test_loss')
    else:
        vis.line(X=np.array([(epoch + 1) / min_epoch]),
                 Y=np.array([loss.item()]),
                 win='test_loss',update='append')
    #
    # if epoch + 1 == 50:
    #     vis.line(X=np.column_stack((np.array([(epoch + 1) / 50]), np.array([(epoch + 1) / 50]))),
    #              Y=np.column_stack(
    #                  (np.array([Test_acc.item()]), np.array([loss.item()]))),
    #              win='test_acc_loss')
    # else:
    #     vis.line(X=np.column_stack((np.array([(epoch + 1) / 50]), np.array([(epoch + 1) / 50]))),
    #              Y=np.column_stack(
    #                  (np.array([Test_acc.item()]), np.array([loss.item()]))),
    #              win='test_acc_loss', update='append')

    print("Epoch %d/%d, test_loss:%.3f, test_acc:%.3f%%(%d/%d)" % (epoch + 1,total_epoch, loss,
                                                                     correct * 100.0 / total, correct, total))
    print("====================================================\n")

    return loss.item(), correct.item() / total


if __name__ == '__main__':
    if use_cuda:
        net.cuda()
    min_epoch = 1
    net.train()
    for i in range(total_epoch):
        train_set, train_label, test_set, test_label = pre_Titanic.get_train_test(pre_Titanic.get_Ti_data())
        train_loss, train_acc = my_train(i, min_epoch, train_set, train_label)
        if (i + 1) % min_epoch == 0:
            test_loss, test_acc = my_test(i, min_epoch, test_set, test_label)
            str1 = str(i + 1) + "\t" + str(train_loss) + "\t" + str(train_acc) + "\t" + str(test_loss) + "\t" + str(
                test_acc)
            vis.text(str1, win='train_info', append=True)