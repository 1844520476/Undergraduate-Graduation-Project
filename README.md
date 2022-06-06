# Undergraduate-Graduation-Project
### 摘要
#### 本文立足于当前图像自动标注领域的两个主要矛盾：迅猛发展的网络架构所需的大规模低噪数据集同数据筛选与标注成本间的矛盾；少样本与模型部署需求同专家标注现状与检验精度要求间的矛盾。为解决减少图片筛选和标注代价与工业罕见案例的小样本分类这两个问题，提出了分别针对目标检测、分类与目标检测加分类任务的三个自动标注模组：
#### （1）基于单阶段迁移目标检测模型的图像标注：本章设计了一种循环标注策略，编写了针对检测任务的自动标注模组，利用Yolov5针对工业数据集进行实验。训练阶段采用迁移学习微调、进化算法调参与优化学习率衰减曲线；预测阶段使用主动学习筛选样本。最终研究了迁移性能的影响因素，验证了基于主动学习的循环标注策略的有效性，并优化了标注模组，实现了按目标类别的图像检索与自动生成数据集等功能，减少了图片的人工筛选和标注代价。
#### （2）基于孪生网络的迁移少样本分类图像标注：本章编写自动分类与标注检验模块，使用表面缺陷纹理分类数据集实验，利用预训练权重进行知识迁移，通过在微调中使用对比学习使孪生网络拥有辨认图像对相似度的能力，取相似度最大的类别作为分类结果。成功研究了迁移学习与对比学习解决少样本工业纹理数据集分类任务的效果并优化，实现了工业小样本分类的目标。 
#### （3）基于目标检测网络与多模态迁移分类网络的复合图像标注系统：基本思想是先用目标检测网络实现定位去冗余与初步分类，再用基于多模态监督信号的迁移视觉模型自定义文本标签实现二次分类。本章基于上述网络模型独立编写并优化复合标注系统，成功实现了包括多目标动态情绪识别与颜色分类等多个自动标注拓展应用。

### 关键词：  迁移学习； 图像自动标注；目标检测；孪生网络；小样本分类； 多模态

### Abstract
#### This paper is based on two major conflicts in the field of automatic image annotation: the conflict between the massive low-noise datasets required by the rapidly developing web architecture and the cost of data screening and annotation; and the conflict between the need for small samples and model deployment and the current state of expert annotation and inspection accuracy requirements. To address the two problems of reducing image screening and labeling costs and classifying small samples for industrial rare cases, three automatic labeling modules are proposed for object detection, classification and object detection plus classification tasks, respectively.
#### (1) Image annotation based on single-stage migrated object detection model: this chapter designs a recurrent annotation strategy, writes an automatic annotation module for the detection task, and conducts experiments using Yolov5 for industrial datasets. The training phase uses migration learning fine-tuning, evolutionary algorithms tuned to optimize the learning rate decay curve; the prediction phase uses active learning to screen samples. Finally, the influencing factors of migration performance are investigated, the effectiveness of the cyclic annotation strategy based on active learning is verified, and the annotation module is optimized to achieve functions such as image retrieval by object category and automatic dataset generation, which reduces the manual screening and annotation cost of images.
#### (2) Migration less sample classification image labeling based on twin networks: This chapter writes an automatic classification and labeling inspection module, experiments with surface defect texture classification dataset, uses pre-training weights for knowledge migration, and makes the twin network possess the ability to recognize the similarity of image pairs by using contrast learning in fine-tuning, and takes the category with the greatest similarity as the classification result. The effectiveness and optimization of migration learning and contrast learning to solve the classification task of small-sample industrial texture dataset is successfully investigated to achieve the goal of industrial small-sample classification. 
#### (3) Composite image labeling system based on object detection network and multimodal migratory classification network: the basic idea is to first use object detection network to achieve localization to remove redundancy and preliminary classification, and then use a migratory vision model based on multimodal supervised signal to customize text labels to achieve secondary classification. This chapter independently writes and optimizes the composite annotation system based on the above network model, and successfully implements several automatic annotation extension applications including multi-object dynamic emotion recognition and color classification.

#### Key Words:  Transfer learning; Automatic; image annotation; Object detection; Twin networks; Few-shot learning; Multi-modal   

### 第三章 循环辅助标注
#### 1.整体
#### ![图3 16](https://user-images.githubusercontent.com/45304468/172265876-b74742c2-1e2b-4553-bfda-896540355213.jpg)
#### 2.单次
#### ![图3 5](https://user-images.githubusercontent.com/45304468/172266019-5fe33848-21ef-46d9-8726-4da12df4ff7e.jpg)


### 第四章 孪生网络小样本分类
#### ![图4 2](https://user-images.githubusercontent.com/45304468/172266232-2861070b-c69c-4f99-96fc-4e152ec7355e.jpg)

#### 纹理数据集相似度对比结果
#### ![图4 3](https://user-images.githubusercontent.com/45304468/172265956-03df071d-5e56-4bc7-83b7-655fde711ad8.jpg)

### 第五章 复合标注系统
#### 多目标人物情绪识别

## <img width="398" alt="image" src="https://user-images.githubusercontent.com/45304468/172266412-72a72add-970e-4105-9b44-e560512e78a4.png">

#### ![图5 5_1](https://user-images.githubusercontent.com/45304468/172266117-a86d2ea1-e1b5-4de0-9cde-afe59da62de2.jpg)

#### 人物肢体与面部情绪识别
#### ![图5 7](https://user-images.githubusercontent.com/45304468/172266192-f0c8d6d4-0c49-4047-b79e-9c684c766ad4.png)




