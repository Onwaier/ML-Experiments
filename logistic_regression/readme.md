## 参考资料

[博客](https://cloud.tencent.com/developer/article/1072473)

## 环境

- torchvision == 0.4.2
- visdom == 0.1.8.9
- numpy == 1.17.4

## 关于Titanic dataset

文件 Titanic_dataset.txt为处理后的文件，Titanic_data_origin.txt为原始文件

- pclass客场等级
```plain
1st:1
2nd:2
3rd:3
```

- survived生还情况

1表示生还，0表示死亡

- name

主要由姓名和称谓组成，也是以,分割，需特殊处理

- age

数据出现错误(0.xxx,NA等)的情况，采用的是用均值代替的方式。

- embarked

数据缺失或者出现多余的信息,对于缺失的补上unknown，最后用均值代替，多余的删除。

```plain
Southampton:0
Cherbourg:1
Queenstown:2
# unknown:3
```
- sex

```plain
male:1
female:2
```

均值：  [1.8928571428571428, 31.622180451127818, 0.3101503759398496, 1.387218045112782]

有问题的行

年龄为NA的行：204，224，244，269，270，473，474，478，492，505，508，522
年龄为0.xxx的行：5，236，375，415，542
登地缺失的行:50
引号位置出错:68,107,171
信息冗余：264,265,310,337


## 数据初始化

4个特征的值分布差异较大，标准化处理，避免不同属性由于数值大小不同对分类结果产生偏差。

```python
num1, num2 = np.shape(data)                              
for j in range(num2 - 1):                                
    mean_val = np.mean(data[:, j])                       
    sta_val = np.std(data[:, j])                         
    data[:, j] = (data[:, j] - mean_val) / sta_val       
```

原始数据集未提供测试集，每个Epoch将所有数据打乱，然后训练集：测试集=4：1
```python
num1, num2 = np.shape(data)       
np.random.shuffle(data)           
train_set = data[:425, :num2 - 1] 
train_label = data[:425, num2 - 1]
test_set = data[425:, :num2 - 1]  
test_label = data[425:, num2 - 1] 
```


## 关于模型

逻辑回归见`models/LinearRegression.py`

```python
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        out = self.linear(x)
        return torch.sigmoid(out) #最后经sigmoid函数归到0到1间
```

## 一些细节

- optimizer
```python
optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.1, momentum=0.9, weight_decay=0)
```

- scheduler(自适应调整学习率)
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
```

## 可视化

使用visdom，观察训练和测试时，acc与loss的变化。
