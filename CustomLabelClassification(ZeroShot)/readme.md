# VLP

## 项目介绍
VLP是一个多模态的视觉语言项目：

1.CLIP:clip原生程序，clip模型是基于自然语言监督信号的迁移视觉模型

2.label:文本标签

3.Input：待识别的图片

4.notebook：自动标注相关调研，gpt-neo的demo，clip的colab实现以及相关的ipynb笔记

5.Weights：模型权重

其他文件夹不一一介绍了：可查看文件里的.txt文件了解文件夹信息

## 模块

1.Detect.py:图片检测程序

2.ZeroShot.py:（针对Cifar10的）zero-shot程序

3.dehaze.py:去雾模块


## ps.如果您觉得配置环境实在不好搞，不要怕，以下是配置好的conda环境（前提是您要安装有Anaconda3）:

clip.zip（百度网盘链接：链接：https://pan.baidu.com/s/1I-Fe4FiuIzFR6fRoa5JcPA?pwd=r03a 
提取码：r03a）

1.解压clip.zip到Anaconda3\envs中

2.将Anaconda3\envs\clip\python.exe配置为项目的python解释器地址

3.运行程序前记得切换环境到clip


## 3.11日更新

1.将循环检测封装到了main.py

2.创建了参数类opt，方便管理参数

3.整合了zero-shot功能到ZeroShot文件夹内，方便添加其他的数据集
