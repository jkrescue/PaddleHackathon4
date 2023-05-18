# <center>基于神经常微分方程的降阶模型</center>

## 1. 项目简介

&emsp;&emsp;该项目是基于飞桨paddlepaddle框架复现论文[Reduced-order Model for Fluid Flows via Neural Ordinary Differential Equations](https://arxiv.org/abs/2102.02248)。该项目的内容是对时间系数的模拟来实现降阶系统（`ROM`）对全阶系统（`FOM`）的重构，具体实现思路如下：

1. 使用LES获取300组全阶系统的模拟数据，[流经圆柱的冯卡门涡街数据集 - 飞桨AI Studio (baidu.com)](https://aistudio.baidu.com/aistudio/datasetdetail/197821)
2. 使用本征正交分解（`POD`）对全阶系统数据分解，选取前8个POD模式
3. 拟设一个分解，假定原时空系统可有分解：时空系数（矩阵）= 时间系数（矩阵） x 空间系数（矩阵）
4. 将时间系数提取出来，输入到一个包含神经微分方程节点的变分自编码器（`VAE`）中，编码器用于提取关键特征，解码器用于还原时间系数
5. 利用还原的参数和原有参数进行比较并且重构时空系统

## 2. 模型介绍

&emsp;&emsp;模型的主体结构是`VAE`的`Encoder-Decoder`结构，编码器和解码器都是由`RNN`构成的，二者中间有一个神经微分方程节点用于处理`Encoder`提取到的关键信息，具体结构可见下图

![结构图](https://github.com/marshall-dteach/Reduced-order-Model-for-Flows-via-Neural-Ordinary-Differential-Equations/blob/main/Figure/Architecture.png)

## 3. 文件结构

~~~
├── cylinderData.pkl           # 数据集
├── VKS_node.py                # Neural ODE模型
├── VKS_lstm.py                # 对比的LSTM模型
├── result.py                  # 输出结果绘制、处理
├── SimpleODEInt.py            # 仿照torchdiffeq写的paddlepaddle处理常微分方程的文件
~~~

## 4. 结果展示

&emsp; &emsp;结果展示分为两部分，一部分是将`Decoder`还原的时间参数与原时间参数对比，查看拟合结果；另一部分是利用还原的时间参数重构时空系统，查看重构效果。

**1. 代码运行2000次，对8个时间参数的拟合结果见下图**

![拟合结果](https://github.com/marshall-dteach/Reduced-order-Model-for-Flows-via-Neural-Ordinary-Differential-Equations/blob/main/Figure/Comparison.png)

**2. 重构时空系统的结果见下图**

![重构结果](https://github.com/marshall-dteach/Reduced-order-Model-for-Flows-via-Neural-Ordinary-Differential-Equations/blob/main/Figure/Result.png)

## 5. 配置需求

|      环境需求      |  显卡需求   |
| :----------------: | :---------: |
| paddlepaddle=2.4.0 | RTX3070 x 1 |

## 6. 快速开始

~~~shell
python VKS_node.py
python VKS_lstm.py
python result.py
~~~

