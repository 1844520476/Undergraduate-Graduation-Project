import datetime
import time
import math

import torch.optim
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ClassifictionAutoNet.vgg16_network import main_vgg
from Resnet import main
from coatnet import *

# 定义训练的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 参数
learning_rate = 0.1  # 优化器中函数参数：学习率(1e-2 = 1x10^(-2)=0.01)
momentum_rate = 0.9
epoch = 5  # 训练轮数
NetMax = 5  # 最大网络编号


# 本地时间
def localTime(define):
    # 打印系统时间
    system_time = datetime.datetime.now().strftime(define)
    return system_time


# 打印当地时间
LocalTime = localTime("%Y%m%d_%H_%M_%S")
print(rf'time:{LocalTime}')

logs_save = rf'Output/10cls/res330'

now_time = localTime('%m_%d_%H_%M_%S')
weightsSavePath = f'weights/train/{now_time}'

print("----------------This is classification by custom network------------------------\n"
      , "______________________总共{}轮__________________".format(epoch))

# tensorboard
writer = SummaryWriter(logs_save)

# 数据集准备
while True:
    dataset_num = int(input(rf'dataset[0]Minst [1]Cifar10 [2]Cifar100:'))
    if dataset_num == 0:
        train_data = torchvision.datasets.MNIST(r"../datasets/data_MNIST", train=True,
                                                transform=torchvision.transforms.ToTensor(), download=True)
        test_data = torchvision.datasets.MNIST(r"../datasets/data_MNIST", train=False,
                                               transform=torchvision.transforms.ToTensor(), download=True)
        break
    elif dataset_num == 1:
        train_data = torchvision.datasets.CIFAR10(r"..\datasets\data_Cifar10", train=True,
                                                  transform=torchvision.transforms.ToTensor(), download=True)
        test_data = torchvision.datasets.CIFAR10(r"..\datasets\data_Cifar10", train=False,
                                                 transform=torchvision.transforms.ToTensor(), download=True)
        break
    elif dataset_num == 2:
        train_data = torchvision.datasets.CIFAR100(r"../datasets/data_Cifar100", train=True,
                                                   transform=torchvision.transforms.ToTensor(), download=True)
        test_data = torchvision.datasets.CIFAR100(r"../datasets/data_Cifar100", train=False,
                                                  transform=torchvision.transforms.ToTensor(), download=True)
        break
    else:
        print(rf'please reinput[0/1/2]:')

# length长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为:{}".format(train_data_size))  # 如果训练数据集的长度为10，则打印 训练数据集的长度为10，{} 被替换为10
print("测试数据集的长度为{}".format(test_data_size))

# Batch size
while True:
    size = int(input('batch size:'))
    if size > 0:
        break
BatchSize_train = size
BatchSize_test = size

# 利用dataloader加载数据集
train_dataloader = DataLoader(train_data, batch_size=BatchSize_train, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BatchSize_test, shuffle=True)


# 选择神经网络
def Input(num):
    # TODO network number
    net_num = int(input(f'\nplease chose network number[0~{num}]:'))
    return net_num


def chose(num_max):
    NetNum = Input(num_max)
    while True:
        if NetNum == 0:
            net = coatnet_0()
            print(f'coatnet_0:{net}')
            return net
        elif NetNum == 1:
            net = coatnet_1()
            print(f'coatnet_1:{net}')
            return net
        elif NetNum == 2:
            net = coatnet_2()
            print(f'coatnet_2:{net}')
            return net
        elif NetNum == 0:
            net = coatnet_3()
            print(f'coatnet_3:{net}')
            return net
        elif NetNum == 4:
            net = coatnet_4()
            print(f'coatnet_4:{net}')
            return net
        elif NetNum == 5:
            net = coatnet_5()
            print(f'coatnet_5:{net}')
            return net
        else:
            print(f'you should choose input 1 ~ {num_max}:')
            NetNum = Input()


newOrnot = input(rf'chose network[y(coatnet)/res(net)/n(vgg16)]:')

if newOrnot == 'n':
    cleste = main_vgg()
    print(rf'cleste model:{cleste}')
elif newOrnot == 'res':
    cleste = main()
    print(rf'resnet:{cleste}')
else:
    cleste = chose(NetMax)
cleste = cleste.to(device)  # 可以写成cleste.to(device0)，仅数据,图片,标准转移之后需要重新赋值

# 损失函数
LossFn = int(input(rf'loss_fn[1]CrossEntropy [2]CTCLoss:'))
while True:
    if LossFn == 1:
        loss_fn = nn.CrossEntropyLoss()
        break
    elif LossFn == 2:
        loss_fn = nn.CTCLoss()
        break
    else:
        print(rf'please reinput loss fuction number')

loss_fn.to(device)

# 设置(初始化)训练网络参数
total_train_step = 1  # 训练次数
total_test_step = 1  # 测试次数

# 计时器
start_time = time.time()
temp = 0

for i in range(epoch):
    print("---------------------------------------第{}轮训练开始---------------------------------------------".format(i + 1))
    step1 = 1

    # 训练步骤开始
    cleste.train()  # 可有可无

    # 优化器
    optimizer = torch.optim.SGD(cleste.parameters(), lr=learning_rate, momentum=momentum_rate)

    total_train_accuracy = 0

    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        # 图像上采样
        upsam = nn.Upsample(scale_factor=7, mode='bilinear')  # 32x32 to 224x224
        if newOrnot == 'y' or 'n':
            imgs = upsam(imgs)
        if dataset_num == 0:
            upsam28 = nn.Upsample(scale_factor=224 / 28, mode='bilinear')
            imgs = upsam28(imgs)
            imgs = imgs.expand(-1, 3, -1, -1)
        targets = targets.to(device)

        # 训练的部分
        outputs = cleste(imgs)
        # TODO 图片shape显示的间隔数
        if total_train_step % 500 == 0:
            print("输入图片的shape为：", imgs.shape)
            # 显示的间隔batch数
            if i % 100 == 0:
                print("\n总第{}次、{}轮的outputs为：{}".format(total_train_step, i + 1, outputs))
        loss = loss_fn(outputs, targets)  # 损失值计算
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        step1 += 1


        def smooth(x):
            x = -x + 1
            x = math.exp(-1 / 10*x) * math.sin(10*x)
            return x


        learning_rate = smooth(learning_rate)

        accuracy_train = (outputs.argmax(1) == targets).sum()
        total_train_accuracy += accuracy_train
        # TODO 显示的间隔batch数
        if total_train_step % 10 == 1:
            print("总第{}次,第{}轮,训练次数：{},loss:{}".format(total_train_step, i + 1, step1,
                                                      loss.item()))  # item的作用是将Tensor数据类型转化为真实数字
        writer.add_scalar("train_loss", loss, total_train_step)

    print("\n整体训练数据集上的准确率：{}%".format(100 * total_train_accuracy / train_data_size))
    writer.add_scalar("train_accuracy", total_train_accuracy / train_data_size, i + 1)

    # 测试步骤开始
    cleste.eval()  # 聊胜于无 ：Sets the module in training mode.This has any effect only on certain modules. See documentations of particular modules for details of their behaviors in training/evaluation mode, if they are affected, e.g. Dropout, BatchNorm, etc.
    total_test_loss = 0.0
    total_accuracy = 0
    with torch.no_grad():
        print("---------------------------------------第{}轮测试开始---------------------------------------------".format(
            i + 1))
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            upsam = nn.Upsample(scale_factor=7, mode='bilinear')  # 32x32 to 224x224
            if newOrnot != 'n':
                imgs = upsam(imgs)
            if dataset_num == 0:
                imgs = upsam(imgs)
                imgs = imgs.expand(-1, 3, -1, -1)
            targets = targets.to(device)
            outputs = cleste(imgs)

            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

            loss = loss_fn(outputs, targets)
            writer.add_scalar("test_loss", loss, total_test_loss)
            total_test_loss += loss.item()

        end_time = time.time()
        time_cost = end_time - start_time
        time_spend = time_cost - temp
        print("\n本轮所耗费时间为{}总耗时为{}".format(time_spend, time_cost))
        temp = time_cost

        print("\n整体测试数据集上的准确率：{}%".format(100 * total_accuracy / test_data_size))
        writer.add_scalar("test_accuracy", total_accuracy / test_data_size, i + 1)

        print("\n训练的总损失值为{}".format(total_test_loss))
        writer.add_scalar("total_test_loss", total_test_loss, i + 1)
        total_test_step += 1

        if i % 10 == 0:
            torch.save(cleste, "{}_{}.pth".format(weightsSavePath, i + 1))
            print("\nModel has benn saved:{}_{}.pth\n".format(weightsSavePath, i + 1))

    writer.close()
