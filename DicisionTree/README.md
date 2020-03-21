### DicisionTree
机器学习第一次作业
### 实验要求
使用决策树对Titanic上人员的存活状况进行预测分类
### 实验环境
python3.6
### 数据预处理
数据集包含550个数据，分为2类，利用每个数据的Pclass、Age、Embarked、Sex 4个属性 来进行分类，  
* Pclass(客场等级）
* Survived(生还状况)，统一规定用1代表生还，用0代表死亡  
* Name(名字)    
* Age(年龄)   
* Embarked(登船港口)  
* Sex(性别)  
![](/image/dataset.png)  
预处理：  
1.对缺失数据及错误数据进行删除处理  
2.将年龄分为baby,child,adult和old  
3.按4:1对数据集进行随即划分为训练集和测试集  
 ![](/image/newdataset.png)  
### 训练结果
##### ID3决策树生成：
![](/image/ID3Tree.png)  
##### CART决策树生成
![](/image/CARDTree.png) 
### 测试结果
|决策树种类|准确率|  
|--|--|
|ID3Tree|78.5% |  
| CARDTree | 80.4% |  

