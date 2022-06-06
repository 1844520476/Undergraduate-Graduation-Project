"""
1.CLIP源代码：https://github.com/OpenAI/CLIP.
2.请在终端中运行下面两行代码（如果是在ipynb环境下运行：请在pip前加上！）
    pip install ftfy regex tqdm
    pip install git+https://github.com/openai/CLIP.git
3.国内镜像源：(1)清华镜像:pip install [The Package You Want to Download] -i https://pypi.tuna.tsinghua.edu.cn/simple
ps.cv2 means opencv-python
4.刘同学于2022.03.13增补
"""
import os
from random import randint
from collections import Counter
import time
import datetime

from prettytable import PrettyTable
import torch
import CLIP
from PIL import Image

from Label.custom import custom_label
from Label.emotional import *
from Label.imagenet import *
from Label.coco128 import *
from Label.cifar10 import *
from dehaze import quwu


# 主程序
def main(WeightsPath, ImgsNameList, InputPathFile):
    print(f'-----------------------------欢迎使用基于自然语言监督信号的迁移视觉模型识别系统--------------------------------')

    # 待测数据集标签选择
    label_exist = (emotion_label_clip, coco128_label, imagenet_label, cifar10_label, custom_label)
    label_dict = {'0': 'emotion', '1': 'coco128', '2': 'imagenet', '3': 'cifar10', '4': 'custom'}
    # 打印所有已存在的标签
    print(f'可选择的【文本】标签：')
    for i in label_dict:
        print(f'[{int(i) + 1}]:{label_dict[i]}_label')
        # label_dict[i] += '_dict'# list indices must be integers or slices, not str(注意此类错误)

    # 选择标签
    def LabelNum():
        while True:
            Label_num = input('请输入对应的【数字编号】：')
            if Label_num == '':
                Label_num = 0
            Label_num = int(Label_num) - 1
            if 0 <= Label_num < len(label_dict):
                Label = label_exist[Label_num]
                print(f'\n您选择的【文本标签详情】如下:\n{Label}')
                return Label, Label_num

    Label, Label_num = LabelNum()
    # 初始化存储概率信息的字典
    dict_prob = {}

    def TextAdd():
        """
        使用add可以方便的对文本信息进行统一修改，
        使格式与训练数据集中的文本图像对更匹配
        """
        text_add = input('\n需要对【文本前后增补】吗？[y/n]:')
        if text_add == 'y':
            print(f'示例：add1 + Label_list[i] + add2（记得打前后空格）')
            add1 = input('add1:')
            add2 = input('add2:')
            return add1, add2
        else:
            add1 = ''
            add2 = ''
            return add1, add2

    # 对原始标签进行增补操作
    add1, add2 = TextAdd()

    # 测试并选择设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 可选模型
    model_dict = {0: 'RN50', 1: 'RN101', 2: 'RN50x4', 3: 'RN50x16', 4: 'RN50x64', 5: 'ViT-B/32', 6: 'ViT-B/16',
                  7: 'ViT-L/14'}

    # 模型选择
    def modelchose():
        while True:
            print(f'\n可选模型：{model_dict}')
            model_num = input('请输入您中意的模型的【数字编号】:')
            if model_num == '':
                model_num = -1
            model_num = int(model_num)
            if 0 <= model_num < len(model_dict):
                model = model_dict[model_num]
                return model

    # 模型选择
    model = modelchose()
    print(f'本次识别使用的【网络模型】为：{model}'
          f'\n.................................网络加载中............................................')

    # 加载模型与预处理
    model, preprocess = CLIP.load(model,
                                  device=device, download_root=WeightsPath)

    # 确定待配对文本信息
    Label_list = list(Label.values())
    # 提前定义列表
    text_input = []

    for i in range(len(Label_list)):
        # 将统一修改后的文本添加到新链表中
        new_label_list = add1 + Label_list[i] + add2
        text_input.append(new_label_list)

    start_time_main = time.time()

    # 图像识别函数
    def detect(img_path):
        # 计时器：开始计时

        # 加载待识别图片
        image = preprocess(Image.open(img_path)). \
            unsqueeze(0).to(device)

        # 读取文本链表
        text = CLIP.tokenize(text_input).to(device)

        # 开始推理过程
        '''
        torch.no_grad() 是一个上下文管理器，被该语句 wrap 起来的部分将不会track 梯度
        即：被此上下文管理器包裹起来计算部分：可以执行计算，但该计算不会在反向传播中被记录
        '''
        with torch.no_grad():
            # 提取文本特征信息到向量空间
            model.encode_image(image)
            # 提取图像特征信息到向量空间
            model.encode_text(text)
            # 计算语义向量与图片特征向量
            logits_per_image, logits_per_text = model(image, text)
            # 计算两者的相似度（计算结果以链表形式按顺序存储）
            probs = logits_per_image.softmax(dim=-1). \
                cpu().numpy()
            probs = probs.tolist()
            # 结束计时
            Time_end_main = time.time() - start_time_main

            # 后处理，打印对应概率
            for i in range(len(text_input)):
                # 以一维list存储了对应text_input的预测概率
                prob = probs[0]
                # 将文本和相似度分别存储为key和value
                prob[i] = round(prob[i] * 100, 3)
                dict_prob[text_input[i]] = prob[i]

            # 最有可能的预测结果
            MAX = sorted(dict_prob,
                         key=dict_prob.get,
                         reverse=True)[0]

            print(f'最有可能的结果是【{MAX}】,'
                  f'有{dict_prob[MAX]:.2f}%的可能性.'
                  f'\n\n(检测过程耗时{Time_end_main:.2f}秒)')  # 预测耗时

            # 画个表格，让输出结果直观漂亮点
            table = PrettyTable(['序号',
                                 '类别',
                                 '预测概率'])

            # 确定最大显示列数
            def length():
                while True:
                    # n = int(input('显示top-n的预测情况[请输入整数以确定n]：'))
                    n = 3
                    if n <= 0:
                        print('请输入大于0的数字')
                    elif n > len(label_exist[Label_num]):
                        print(f'请输入小于文本标签长度的数字'
                              f'(标签长度为【{len(label_exist[Label_num])}】)')
                    else:
                        print(f'将显示top-{n}的预测概率')
                        return n
                        break

            # 按value的大小排序
            table_length = length()

            # 按概率大小的顺序翻译
            dict_cn = {}
            for i in range(table_length):
                prob_max_i = sorted(dict_prob,
                                    key=dict_prob.get,
                                    reverse=True)[i]
                probability = str(dict_prob[prob_max_i]) + '%'

                # TODO 将英文标签转化为中文
                label_cn_dict = {0: emotional_dict, 1: '', 2: '', 3: cifar10_dict, 4: ''}  # 标签的中文翻译（以字典形式存储）
                label_cn_dict = label_cn_dict[Label_num]  # 根据标签选择确定对应的中文翻译字典
                if label_cn_dict != '':
                    # 将翻译后的key映射回翻译前的key
                    text_i = text_input.index(prob_max_i)
                    prob_max_i = Label_list[text_i]

                    prob_max_i_cn = label_cn_dict[prob_max_i] + '(' + prob_max_i + ')'
                    dict_cn[prob_max_i] = prob_max_i_cn
                else:
                    prob_max_i_cn = prob_max_i
                    dict_cn[prob_max_i] = prob_max_i_cn

                # 画表格
                table.add_row([i + 1,
                               prob_max_i_cn,
                               probability])

            # 打印预测概率top-n表格
            print(f'{table}')
        return probs[0]

    PilotEmoDict = {}
    MaxPilotEmoDict = {}

    for i in ImgsNameList:
        # 图片地址
        imgsnameList = i
        imgsnameList = f'{InputPathFile}/{imgsnameList}'
        if os.path.exists(imgsnameList):
            PilotEmo = detect(imgsnameList)
            print(f'\n针对【图片:{imgsnameList}】的【识别结果】为:')
            PilotEmoDict[f'{imgsnameList}'] = PilotEmo
            MaxPilotEmoDict[f'{imgsnameList}'] = PilotEmo.index(max(PilotEmo))

    return MaxPilotEmoDict, PilotEmoDict


# 本地时间
def localTime(define):
    # 打印系统时间
    system_time = datetime.datetime.now().strftime(define)
    return system_time


# 创建文件夹
def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        print("---  OK  ---")

    else:
        print("---  There is this folder!  ---")


# 执行主程序
if __name__ == '__main__':
    while True:
        InputDir = 'Input'

        #检测文件夹所在地址
        def InputPath():
            # 载入地址
            W = input(f'请输入待检测的文件夹：')
            while True:
                if len(W) == 0:
                    W = input(f'输入为空，请输入{InputDir}中存在的文件夹')
                elif os.path.exists(f'{InputDir}/{W}'):
                    return W
                else:
                    W = input(f'输入不存在，请输入{InputDir}中存在的文件夹')

        W = InputPath()
        Start_time = time.time()
        InputPath_file = rf'{InputDir}/{W}'
        imgs_nameList = os.listdir(InputPath_file)
        # 存储地址
        # TODO 1.存储地址
        now_time = localTime('%m_%d_%H_%M_%S')
        Output_path = f'Output/{W}/{now_time}'
        # 情绪权重
        # TODO 2.情绪权重
        # W_emotion = float(input(f'情绪权重：'))
        # W = str(int(float(W_emotion) * 100))
        # 创建对应的文件夹
        mkdir(f'{Output_path}')
        # mkdir(f'{Output_path}/{W}_max')
        # mkdir(f'{Output_path}/{W}')
        # 本地时间
        system_time = localTime('month:%m day:%d %H:%M')
        print(f'现在是北京时间：{system_time}')
        # 模型存储地址
        weightspath = r'weights'
        # 主程序
        # TODO 4.待检测文件夹数目
        # FileNum = int(60)
        # Train_i = 1

        # while Train_i < FileNum + 1:
        MAX_EMO = []
        # 打印待检测文件夹地址
        print(f'\n待检测的文件夹地址：{InputPath_file}')
        MaxEmo, Emo = main(weightspath, imgs_nameList, InputPath_file)
        print(f'MaxEmo:{MaxEmo}')
        print(f'\nEmo:{Emo}')
        # 写入检测数据到.txt
        file1 = open(f'{Output_path}/{W}_quan.txt', 'a+')
        file2 = open(f'{Output_path}/{W}_max.txt', 'a+')
        file3 = open(f'{Output_path}/{W}.txt', 'a+')
        for i in Emo.keys():
            # 存储量化数据
            Emo_i = ' '.join(str(i) for i in Emo[i])
            file1.write(f'{i} {Emo_i}' + '\n')
            # 存储最大类别
            # Emo[i][2] = Emo[i][2] * W_emotion
            file2.write(f'{Emo[i].index(max(Emo[i]))}' + '\n')
            MAX_EMO.append(Emo[i].index(max(Emo[i])))
        file1.close()
        file2.close()
        # print(f'\n修正后的类别结果：{MAX_EMO}')
        # 打印出现次数
        MAX_EMO = ','.join(str(i) for i in MAX_EMO)
        num1 = MAX_EMO.split(',')
        num2 = {}
        for i in num1:
            if i not in num2:
                num2[i] = 1
            else:
                num2[i] = num2[i] + 1
        num3 = Counter(num2).most_common()  # 排序
        print('出现的次数按照从大到小：')
        num2 = dict(num3)  # 把列表转换成字典
        for key, v in num2.items():
            '''
                        for key, v in num2.items():
                        # label = f'{Label_global}[{key}]'
                            label = custom_label[{key}]
                            print({f'{label}' + ':' + '出现' + str(v) + '次')
                            file3.write(f'{label}' + ':' + '出现' + str(v) + '次' + '\n')
                        '''
            # TODO label file error : 应该把标签字典聚合成list
            detect_result = emotion_label_clip[f'{key}'] + ':' + '出现' + str(v) + '次'
            print(detect_result)
            file3.write(detect_result + '\n')
        file3.close()
        # Train_i += 1
        End_time = time.time() - Start_time
        print(f'耗时：{End_time:.2f}s'
              f'检测结果存储地址：{Output_path}')
        # 退出与否
        exit = input('是否退出系统:[y/n]')
        if exit == 'y':
            print(f'-----------------------------期待您再次使用[自定义标签]自动标注系统，再见>_<！--------------------------------')
            break
        else:
            print('即将返回主界面')
