import os
import numpy as np
import h5py
import pandas as pd


def deal_each(s):
    return s.strip()

def get_Ti_data():
    Ti_data_path = os.path.join('data', 'Titanic_dataset_origin.txt')
    if not os.path.exists(Ti_data_path):
        os.mkdir(Ti_data_path)

    Ti_data = []
    embarked_list = {'Southampton': 0, 'Cherbourg': 1, 'Queenstown': 2, 'unknown': 3}
    # 计数
    cnt = 0

    # 打开txt文件
    f = open(Ti_data_path)

    for line in f.readlines():

        if cnt == 0:  # 跳过第一行
            cnt = cnt + 1
            continue
        str_list = line.split(',')

        str_list = list(map(deal_each, str_list))
        line_data = []  # 存储pclass(1), age(5), embarked(6), sex(7)

        # 处理pclass
        str_list[1] = eval(str_list[1])
        if str_list[1] == '1st':
            line_data.append(1)
        elif str_list[1] == '2nd':
            line_data.append(2)
        elif str_list[1] == '3rd':
            line_data.append(3)

        idx = 4
        while str_list[idx][0].isdigit() == False and str_list[idx] != 'NA':
            idx = idx + 1
        # 处理age
        if str_list[idx] == 'NA':
            line_data.append(-1)
        else:
            line_data.append(int(float(str_list[idx])))

        # 处理embarked
        idx = idx + 1
        str_list[idx] = eval(str_list[idx])
        # if str_list[idx] not in embarked_list.keys():
        #     embarked_list[str_list[idx]] = len(embarked_list)
        line_data.append(embarked_list[str_list[idx]])

        # 处理sex

        idx = idx + 1
        str_list[idx] = eval(str_list[idx])
        if str_list[idx] == 'male':
            line_data.append(1)
        elif str_list[idx] == 'female':
            line_data.append(2)

        # 处理survived
        line_data.append(int(str_list[2]))  # 1表示生还，0表示死亡

        Ti_data.append(line_data)

        # 处理survived



    # print(np.shape(Ti_data))
    data = np.array(Ti_data)

    # 去中心化和标准化
    num1, num2 = np.shape(data)
    mean_vals = []
    for j in range(num2 - 1):
        mean_val = np.mean(data[:, j])
        sta_val = np.std(data[:, j])
        data[:, j] = 1.0 *  (data[:, j] - mean_val) / sta_val
        mean_vals.append(mean_val)
    # print(mean_vals)
    return data

def get_train_test(data):
    num1, num2 = np.shape(data)
    np.random.shuffle(data)
    train_set = data[:425, :num2 - 1]
    train_label = data[:425, num2 - 1]
    test_set = data[425:, :num2 - 1]
    test_label = data[425:, num2 - 1]
    return train_set, train_label, test_set, test_label

if __name__ == '__main__':

    Ti_data = get_Ti_data()
    train_set, train_label, test_set, test_label = get_train_test(Ti_data)
    print(np.shape(Ti_data))



