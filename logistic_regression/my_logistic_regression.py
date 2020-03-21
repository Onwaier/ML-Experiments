import numpy as np
import preprocess_Titanic as pre_Titanic
import visdom

vis = visdom.Visdom(env='my_Logistic_Regression_model')

# 自定义sigmoid函数
def sigmoid(z):
    a = 1/(1+np.exp(-z))
    return a

# init w和b(传入维度)
def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0
    return w,b

# 前向传播
def propagate(w, b, X, Y):
    """
    传参:
    w -- 权重, shape： (num_px * num_px * 3, 1)
    b -- 偏置项, 一个标量
    X -- 数据集，shape： (num_px * num_px * 3, m),m为样本数
    Y -- 真实标签，shape： (1,m)

    返回值:
    cost， dw ，db，后两者放在一个字典grads里
    """
    #获取样本数m：
    m = X.shape[1]

    # 前向传播 ：
    A = sigmoid(np.dot(w.T,X)+b)    #调用前面写的sigmoid函数
    cost = -(np.sum(Y*np.log(A)+(1-Y)*np.log(1-A)))/m    # 交叉熵损失函数

    # 反向传播：
    dZ = A-Y
    dw = (np.dot(X,dZ.T))/m
    db = (np.sum(dZ))/m

    #返回值：
    grads = {"dw": dw,
             "db": db}

    return grads, cost

# 优化器
def optimize(w, b, train_set, train_label, learning_rate):

    # 用propagate计算出每次迭代后的cost和梯度：


    # print(train_set.shape)
    # print(train_label.shape)
    grads, train_loss = propagate(w,b,train_set,train_label)
    dw = grads["dw"]
    db = grads["db"]


    # 用上面得到的梯度来更新参数：
    w = w - learning_rate*dw
    b = b - learning_rate*db


    #迭代完毕，将最终的各个参数放进字典，并返回：
    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}
    return params, grads, train_loss

# 预测函数
def predict(w,b,X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))

    A = sigmoid(np.dot(w.T,X)+b)
    for  i in range(m):
        if A[0,i]>0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0

    return Y_prediction

# 绘制acc-loss曲线
def my_plot(epoch, mini_epoch, train_loss, train_acc, test_loss, test_acc):
    if epoch + 1 == mini_epoch:
        vis.line(X=np.array([(epoch + 1) / mini_epoch]),
                 Y=np.array([train_acc]),
                 win='train_acc')
    else:
        vis.line(X=np.array([(epoch + 1) / mini_epoch]),
                 Y=np.array([train_acc]),
                 win='train_acc', update='append')

    if epoch + 1 == mini_epoch:
        vis.line(X=np.array([(epoch + 1) / mini_epoch]),
                 Y=np.array([train_loss]),
                 win='train_loss')
    else:
        vis.line(X=np.array([(epoch + 1) / mini_epoch]),
                 Y=np.array([train_loss]),
                 win='train_loss', update='append')

    if epoch + 1 == mini_epoch:
        vis.line(X=np.array([(epoch + 1) / mini_epoch]),
                 Y=np.array([test_acc]),
                 win='test_acc')
    else:
        vis.line(X=np.array([(epoch + 1) / mini_epoch]),
                 Y=np.array([test_acc]),
                 win='test_acc', update='append')

    if epoch + 1 == mini_epoch:
        vis.line(X=np.array([(epoch + 1) / mini_epoch]),
                 Y=np.array([test_loss]),
                 win='test_loss')
    else:
        vis.line(X=np.array([(epoch + 1) / mini_epoch]),
                 Y=np.array([test_loss]),
                 win='test_loss', update='append')

# 定义logistic模型
def logistic_model(mini_epoch = 20, learning_rate=0.1,total_epoch=500):
    #获特征维度，初始化参数：
    train_set, train_label, test_set, test_label = pre_Titanic.get_train_test(pre_Titanic.get_Ti_data())
    dim = train_set.shape[1]
    W,b = initialize_with_zeros(dim)

    ave_train_acc = 0.0
    ave_train_loss = 0.0
    ave_test_acc = 0.0
    ave_test_loss = 0.0

    for i in range(total_epoch):
        if (i + 1) % mini_epoch == 1:
            ave_train_acc = ave_train_loss = ave_test_acc = ave_test_loss = 0.0
        train_set, train_label, test_set, test_label = pre_Titanic.get_train_test(pre_Titanic.get_Ti_data())
        train_set = train_set.T
        train_label = train_label.T
        test_set = test_set.T
        test_label = test_label.T

        #梯度下降，迭代求出模型参数：
        params,grads,train_loss = optimize(W, b, train_set, train_label, learning_rate)
        W = params['w']
        b = params['b']

        #用学得的参数进行预测：
        prediction_train = predict(W, b, train_set)
        prediction_test = predict(W, b, test_set)

        #计算准确率，分别在训练集和测试集上：
        train_acc = 1 - np.mean(np.abs(prediction_train - train_label))
        test_acc = 1 - np.mean(np.abs(prediction_test - test_label))

        A = sigmoid(np.dot(W.T,test_set)+b)    #调用前面写的sigmoid函数
        test_loss = -(np.sum(test_label*np.log(A)+(1-test_label)*np.log(1-A))) / test_set.shape[1]    # 交叉熵损失函数

        print("Epoch %d/%d, train_loss:%.3f, train_acc:%.3f%%" % (i + 1, total_epoch, train_loss,
                                                                         train_acc * 100.0))
        print("Epoch %d/%d, test_loss:%.3f, test_acc:%.3f%%" % (i + 1, total_epoch, test_loss,
                                                                  test_acc * 100.0))
        print("====================================================\n")

        ave_train_acc += train_acc
        ave_train_loss += train_loss
        ave_test_acc += test_acc
        ave_test_loss += test_loss
        if (i + 1) % mini_epoch == 0:
            ave_train_loss = ave_train_loss / mini_epoch * 100
            ave_train_acc = ave_train_acc / mini_epoch * 100
            ave_test_loss = ave_test_loss  / mini_epoch * 100
            ave_test_acc = ave_test_acc  / mini_epoch * 100
            my_plot(i, mini_epoch, ave_train_loss, ave_train_acc, ave_test_loss, ave_test_acc)
   #为了便于分析和检查，我们把得到的所有参数、超参数都存进一个字典返回出来：
    # d = {"costs": costs,
    #      "Y_prediction_test": prediction_test ,
    #      "Y_prediction_train" : prediction_train ,
    #      "w" : W,
    #      "b" : b,
    #      "learning_rate" : learning_rate,
    #      "num_iterations": num_iterations,
    #      "train_acy":accuracy_train,
    #      "test_acy":accuracy_test
    #     }
    # return d

if __name__ == '__main__':

    logistic_model(mini_epoch = 20, total_epoch=8000, learning_rate=0.01)