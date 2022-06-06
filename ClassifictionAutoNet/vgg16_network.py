import torch
import torch.nn as nn


class VGG16(nn.Module):
    def __init__(self, nums):
        super(VGG16, self).__init__()
        self.nums = nums
        vgg = []

        # 第一个卷积部分
        # 112, 112, 64
        vgg.append(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU())
        vgg.append(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU())
        vgg.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # 第二个卷积部分
        # 56, 56, 128
        vgg.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU())
        vgg.append(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU())
        vgg.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # 第三个卷积部分
        # 28, 28, 256
        vgg.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU())
        vgg.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU())
        vgg.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU())
        vgg.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # 第四个卷积部分
        # 14, 14, 512
        vgg.append(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU())
        vgg.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU())
        vgg.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU())
        vgg.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # 第五个卷积部分
        # 7, 7, 512
        vgg.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU())
        vgg.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU())
        vgg.append(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1))
        vgg.append(nn.ReLU())
        vgg.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # 将每一个模块按照他们的顺序送入到nn.Sequential中,输入要么事orderdict,要么事一系列的模型，遇到上述的list，必须用*号进行转化
        self.main = nn.Sequential(*vgg)

        # 全连接层
        classfication = []
        # in_features四维张量变成二维[batch_size,channels,width,height]变成[batch_size,channels*width*height]
        classfication.append(
            nn.Linear(in_features=512 * 7 * 7, out_features=4096))  # 输出4096个神经元，参数变成512*7*7*4096+bias(4096)个
        classfication.append(nn.ReLU())
        classfication.append(nn.Dropout(p=0.5))
        classfication.append(nn.Linear(in_features=4096, out_features=4096))
        classfication.append(nn.ReLU())
        classfication.append(nn.Dropout(p=0.5))
        classfication.append(nn.Linear(in_features=4096, out_features=self.nums))

        self.classfication = nn.Sequential(*classfication)

    def forward(self, x):
        feature = self.main(x)  # 输入张量x
        feature = feature.view(x.size(0), -1)  # reshape x变成[batch_size,channels*width*height]
        result = self.classfication(feature)
        return result

def main_vgg():
    cls_num = int(input(f'classes number:'))
    net = VGG16(cls_num)
    return  net

if __name__ == '__main__':
    def vgg_create(nums):
        x = torch.rand(size=(1, 3, 224, 224))
        vgg16 = VGG16(nums)
        out = vgg16(x)
        print(f'1.1 vgg16:{vgg16}'
              f'1.2 x:{x}'
              f'\n2.x_out:{out}'
              f'\n3.x_out.shape:{out.shape}')

    while True:
        classes_nums = int(input(f'classes number:'))
        vgg_create(classes_nums)
