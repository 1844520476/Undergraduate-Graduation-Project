"""
注意数据集下载地址，因为是相对路径，要随data_in()函数所在位置的不同而修改
"""
import time
import datetime
import torch
import torchvision
from torch.utils.data import DataLoader
from Option import opt
import CLIP
import torch.optim
from prettytable import PrettyTable
from torch import nn
from Label.emotional import *
from Label.imagenet import *
from Label.coco128 import *
from Label.cifar10 import *

# 主程序
def main_zeroshot(opt):
    WeightsPath, Epoch, Number, TestDataloader, Label_num = opt.weightspath, opt.epoch, opt.number, opt.test_dataloader, opt.Label_num
    print(f'-----------------------------欢迎使用基于自然语言监督信号的迁移视觉模型识别系统--------------------------------')

    # 打印系统时间
    define = '%m月%d日 %H时%M分'
    system_time = datetime.datetime.now().strftime(define)
    print(f'现在是北京时间：{system_time}')

    # 待测数据集标签选择
    label_exist = (emotion_label, coco128_label, imagenet_label, cifar10_label)
    label_dict = {'0': 'emotion', '1': 'coco128', '2': 'imagenet', '3': 'cifar10'}

    # 打印所有已存在的标签
    print(f'可选择的【文本】标签：')
    for i in label_dict:
        print(f'[{int(i) + 1}]:{label_dict[i]}_label')
        # label_dict[i] += '_dict'# list indices must be integers or slices, not str(注意此类错误)

    # 选择标签
    Label = label_exist[Label_num]
    print(f'\n您选择的【文本标签详情】如下:\n[{Label_num + 1}]:{Label}')

    # 初始化存储概率信息的字典
    dict_prob = {}

    def TextAdd():
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
            model_num = int(input('请输入您中意的模型的【数字编号】:'))
            if 0 <= Label_num < len(model_dict):
                model = model_dict[model_num]
                return model
                break

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
    '''
    使用add可以方便的对文本信息进行统一修改，
    使格式与训练数据集中的文本图像对更匹配
    '''
    for i in range(len(Label_list)):
        # 将统一修改后的文本添加到新链表中
        new_label_list = add1 + Label_list[i] + add2
        text_input.append(new_label_list)

    # 图像识别函数
    def detect(img, img_n):
        image = img
        # 图片地址
        # img_path = img

        # 计时器：开始计时
        start_time = time.time()

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
                cpu().numpy().tolist()
            # 结束计时
            Time = time.time() - start_time

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

            # 画个表格，让输出结果直观漂亮点
            table = PrettyTable(['序号',
                                 '类别',
                                 '预测概率'])

            # 确定最大显示列数
            def length():
                while True:
                    n = 5
                    if n <= 0:
                        print('请输入大于0的数字')
                    elif n > len(label_exist[Label_num]):
                        print(f'请输入小于文本标签长度的数字'
                              f'(标签长度为【{len(label_exist[Label_num])}】)')
                    else:
                        print(f'\n将显示top-{n}的预测概率')
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

                # 将英文标签转化为中文
                label_cn_dict = {0: emotional_dict, 1: '', 2: '', 3: cifar10_dict}  # 标签的中文翻译（以字典形式存储）
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

            print(f'针对【{img_n}】的【识别结果】为:')
            print(f'\n最有可能的结果是【{MAX}】,'
                  f'有{dict_prob[MAX]:.2f}%的可能性.'
                  # 预测耗时
                  f'(检测过程耗时{Time:.2f}秒)')

            return probs

    i = 0
    acc = []
    while Epoch > 0:
        # 测试轮数
        i += 1
        # 计时器
        start_time = time.time()
        temp = 0
        # 测试步骤开始
        total_accuracy = 0
        print("---------------------------------------第{}轮测试开始---------------------------------------------"
              .format(i))
        with torch.no_grad():
            img_num = 0
            for data in TestDataloader:
                img_num += 1
                imgs, targets = data
                imgs = imgs.to(device)
                """
                转置卷积失败报错
                RuntimeError: The size of tensor a (2) must match the size of tensor b (50) at non-singleton dimension 1

                """
                # 图像上采样
                upsam = nn.Upsample(scale_factor=7, mode='bilinear')
                imgs = upsam(imgs)

                targets = targets.to(device)

                # imgs已经是tensor数据类型
                outputs = detect(imgs, img_num)

                outputs = outputs[0]
                out = outputs.index(max(outputs))
                tar = targets.tolist()
                tar = tar[0]
                # accuracy = (out == tar).sum()
                # total_accuracy += accuracy
                if out == tar:
                    total_accuracy += 1

                if img_num >= Number:
                    break

        end_time = time.time()
        time_cost = end_time - start_time
        time_spend = time_cost - temp
        print(f"\n本轮所耗费时间为{time_spend:.3f}秒")
        epoch_accuracy = 100 * total_accuracy / Number
        print("\n第[{}]轮：整体测试数据集上的zero-shot准确率为{}%".format(i, epoch_accuracy))
        acc.append(epoch_accuracy)
        Epoch -= 1

    return acc

#加载数据集
def data_in(Opt):
        # 数据集准备
        test_data = torchvision.datasets.CIFAR10(r"Dataset\Cifar10", train=False,
                                                 transform=torchvision.transforms.ToTensor(), download=True)

        # length长度
        test_data_size = len(test_data)
        print("测试数据集的长度为{}".format(test_data_size))

        # 利用dataloader加载数据集
        test_dataloader = DataLoader(test_data, batch_size=1)
        print(f'数据已加载完成')

        # 确定epoch与number
        Opt.number = int(input(f'请输入number[检测图片数量]：'))
        Opt.epoch = int(input(f'请输入epoch[检测轮数]：'))
        # 导入数据集名称
        Opt.DatasetName = 'CIFAR10'
        # 对应标签选择
        Opt.Label_num = 3
        # 模型存储地址
        Opt.weightspath = r'..\weights'
        # 数据集加载
        Opt.test_dataloader = test_dataloader
        return test_dataloader, Opt

if __name__ == '__main__':
    # 初始化参数类
    Opt = opt()
    # 加载数据与参数
    test_dataloader, Option = data_in(Opt)
    accuracy = main_zeroshot(Option)
    print(f'accuracy:{accuracy}')
