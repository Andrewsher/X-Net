# X-Net
[X-Net: Brain Stroke Lesion Segmentation Based on Depthwise Separable Convolution and Long-range Dependencies (MICCAI 2019)](https://arxiv.org/abs/1907.07000)
# 作者
Kehan Qi, Hao Yang, Cheng Li, Zaiyi Liu, Meiyun Wang, Qiegen Liu, and Shanshan Wang
# 项目简介
## 1. 功能
采用X-Net实现对ATLAS数据集的图像分割
## 2. 性能
|Dice|IoU|Precision|Recall|Number of Parameters|
|-----|-----|-----|-----|-----|
|0.4867|0.3723|0.6000|0.4752|15.1M|
## 3. 使用数据集
包括数据集名称、来源。如果不使用数据集，则留空。
数据集：ATLAS数据集[1]，包含229个case，采用5折交叉验证。数据采用[这里](https://github.com/Andrewsher/ATLAS-dataset-generate-h5file)所示的方法进行预处理得到h5文件。
[1] Liew, Sook-Lei, et al. "A large, open source dataset of stroke anatomical brain images and manual lesion segmentations." Scientific data 5 (2018): 180011.
# 运行环境与依赖
|类别|名称|版本|
|-----|-----|-----|
|os|ubuntu|16.04|
|深度学习框架|Keras|2.2.4|
|深度学习框架|tensorflow|1.14.0|
|机器学习库|scikit-learn|0.19.1|
|python函数库|pandas|0.20.3|
|处理h5文件的库|h5py|2.7.0|

# 输入与输出
|名称|说明|
|输入|单通道灰度图，值域为0-1，大小为224x192|
|输出|标签。0表示背景，1表示病变|

# 运行方式
在main.py中修改与超参数相关的行（即第16-20行），然后在命令行中执行如下的命令：
```shell
python main.py
```
