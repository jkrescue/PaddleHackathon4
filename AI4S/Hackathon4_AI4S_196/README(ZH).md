# 用户文档
<u>_wanghanweibnds2015@gmail.com_</u>

---

[English](./README(EN).md) | 简体中文

## 简介

---

使用PINN进行科学计算时通常需要部分有监督数据,本套件的开发旨在提取CFD计算结果并转化为PaddleScience框架可读取的形式.
目前支持OpenFOAM的结果数据.

## 从此开始

---

### 下载依赖包

---

`pip install -r dependencies/requirements.txt`

### 运行示例

---

`demo`文件夹中收纳了两个简单示例.
本说明用到了OpenFOAM icoFoam 求解器计算的顶盖驱动流算例.
CFD的计算结果存储于`demo/OpenFOAMCavity`.

在Python环境中运行demo的代码.

了解基础用法和数据格式:
```commandline
cd demo
python read_cavity.py
```

向PINN程序植入CFD数据:
```commandline
cd demo
python insert_from_cavity.py
```

## API参阅
[![Documentation Status](https://img.shields.io/badge/API参阅-blue.svg)](./doc/API-reference.md)