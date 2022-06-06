# .to(device)
# 使用CPU训练
# Device = torch.device("cpu")
# 使用GPU训练
# 1.仅一张显卡
# torch.device("cuda")
# 2.如果有多张显卡的情况
# torch.device("cuda:0")
# torch.device("cuda:1")

# 使用GPU训练：CUDA
# 网络模型：数据（输入，标注）；损失函数；cuda

import time
import torch.optim
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from coatnet import *

# 注意：tensorboard打开没东西可能是因为conda环境不正确

# 定义训练的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 参数
learning_rate = 1e-2  # 优化器中函数参数：学习率(1e-2 = 1x10^(-2)=0.01)
momentum_rate = 0.9
epoch = 100  # 训练轮数
size = 16
BatchSize_train = size
BatchSize_test = size
NetMax = 5#最大网络编号

# 当地时间
Time = time.localtime()
LocalTime = time.strftime("%Y%m%d_%H_%M_%S", Time)
logs_save = rf'Output/100cls/0328'
pt_save_name = 'Output/pt/100'

print(
    "-------------------------------------------------------------------This is classification by custom network------------------------------------------------------------------------------------------\n"
    ,
    "_______________________________________________________总共{}轮_________________________________________________________________".format(
        epoch))

# tensorboard
writer = SummaryWriter(logs_save)

# 数据集准备
train_data = torchvision.datasets.CIFAR100("../datasets/data_Cifar100", train=True,
                                          transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR100("../datasets/data_Cifar100", train=False,
                                         transform=torchvision.transforms.ToTensor(), download=True)

# length长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为:{}".format(train_data_size))  # 如果训练数据集的长度为10，则打印 训练数据集的长度为10，{} 被替换为10
print("测试数据集的长度为{}".format(test_data_size))

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
        if  NetNum == 0:
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

#加载神经网路

class Cleste(nn.Module):
    def __init__(self):
        super(Cleste, self).__init__()
        self.numClasses = 0
        self.model = nn.Sequential(

            nn.Conv2d(3, 16, 7, 1, 3),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),  # 4x4
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.MaxPool2d(2),  # 2x2,此时最优

            nn.Flatten(),
            nn.Linear(512, 200),  # 记得更改输入通道数
            # 我好像没有激活函数欸，没有非线性单元怎么还成功了
            nn.Linear(200,100)
        )

    def forward(self, x):
        #NumClasses = self.numClasses
        x = self.model(x)
        return x

newOrnot = input(rf'coatnet[y/n]:')
if newOrnot == 'n':
    cleste = Cleste()
    #cleste.numClasses = int(input(rf'num_classes'))
    print(rf'cleste model:{cleste}')
else:
    cleste = chose(NetMax)
cleste = cleste.to(device)  # 可以写成cleste.to(device0)，仅数据,图片,标准转移之后需要重新赋值

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)
# 优化器
optimizer = torch.optim.SGD(cleste.parameters(), lr=learning_rate, momentum=momentum_rate)

# 设置训练网络参数
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

    total_train_accuracy = 0

    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        # 图像上采样
        upsam = nn.Upsample(scale_factor=7, mode='bilinear')#32x32 to 224x224
        if newOrnot == 'y':
            imgs = upsam(imgs)
        targets = targets.to(device)

        # 训练的部分
        outputs = cleste(imgs)
        if total_train_step % 500 == 0:
            print("输入图片的shape为：", imgs.shape)
            if i % 10 == 0:
                print("\n总第{}次、{}轮的outputs为：{}".format(total_train_step, i+1, outputs))
        loss = loss_fn(outputs, targets)  # 损失值计算
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        step1 += 1

        accuracy_train = (outputs.argmax(1) == targets).sum()
        total_train_accuracy += accuracy_train

        if total_train_step % 1 == 0:
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
            #imgs = imgs.to(device).convert('RGB')
            if newOrnot == 'y':
                imgs = upsam(imgs)
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
        writer.add_scalar("total_test_loss", total_test_loss, i+1)
        total_test_step += 1

        if i % 10 == 9:
            torch.save(cleste, "{}_{}.pth".format(pt_save_name, i + 1))
            print("\nModel has benn saved:{}_{}.pth\n".format(pt_save_name, i + 1))

    writer.close()
