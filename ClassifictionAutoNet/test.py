import time

import cv2
from coatnet import *
import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def transform_convert(img_tensor, transform):
    """
    param img_tensor: tensor
    param transforms: torchvision.transforms
    """
    if 'Normalize' in str(transform):
        normal_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform.transforms))
        mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        img_tensor.mul_(std[:, None, None]).add_(mean[:, None, None])

    img_tensor = img_tensor.transpose(0, 2).transpose(0, 1)  # C x H x W  ---> H x W x C

    if 'ToTensor' in str(transform) or img_tensor.max() < 1:
        img_tensor = img_tensor.detach().numpy() * 255

    if isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor.numpy()

    if img_tensor.shape[2] == 3:
        img = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
    elif img_tensor.shape[2] == 1:
        img = Image.fromarray(img_tensor.astype('uint8')).squeeze()
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_tensor.shape[2]))

    return img


def test_coatnet_0(img):
    net = coatnet_0().cuda()
    out = net(img)
    # print(f'out[coatnet_0]:{out}')
    return out


def test_coatnet_1(img):
    net = coatnet_1().cuda()
    out = net(img)
    return out


def test_coatnet_2(img):
    net = coatnet_2().cuda()
    out = net(img)
    return out


def test_coatnet_3(img):
    net = coatnet_3().cuda()
    out = net(img)
    return out


def test_coatnet_4(img):
    net = coatnet_4().cuda()
    out = net(img)
    return out


def test_coatnet_5(img):
    net = coatnet_5().cuda()
    out = net(img)
    return out


# custom network
def test_CoAtNet(img):
    num_blocks = [2, 2, 3, 5, 2]  # L
    channels = [64, 96, 192, 384, 768]  # D
    block_types = ['C', 'T', 'T', 'T']  # 'C' for MBConv, 'T' for Transformer

    net = CoAtNet((224, 224), 3, num_blocks, channels, block_types=block_types).cuda()
    out = net(img)
    # print(f'out[CoAtNet]:{out}')
    return out


def Input(num):
    # TODO network number
    net_num = int(input(f'\nplease chose network number[0~{num}]:'))
    return net_num


def chose(Img, num_max):
    NetNum = Input(num_max)
    while True:
        if NetNum == 114154:
            result = None
            return result, NetNum
        elif NetNum == 0:
            result = test_coatnet_0(Img)
            return result, NetNum
        elif NetNum == 1:
            result = test_coatnet_1(Img)
            return result, NetNum
        elif NetNum == 2:
            result = test_coatnet_2(Img)
            return result, NetNum
        elif NetNum == 3:
            result = test_coatnet_3(Img)
            return result, NetNum
        elif NetNum == 4:
            result = test_coatnet_4(Img)
            return result, NetNum
        elif NetNum == 5:
            result = test_coatnet_5(Img)
            return result, NetNum
        elif NetNum == 6:
            result = test_CoAtNet(Img)
            return result, NetNum
        else:
            print(f'you should choose input 1 ~ {num_max}:')
            NetNum = Input()


def main(image):
    max = 6
    Output, netNum = chose(image, max)
    #TODO  softmax
    sm = nn.Softmax(dim=1)
    Output = sm(Output)
    return Output


if __name__ == '__main__':
    # 打印当地时间
    Time = time.localtime()
    LocalTime = time.strftime("%Y%m%d_%H_%M_%S", Time)
    print(f'现在是北京时间{LocalTime}')
    # main function
    img = torch.randn(1, 3, 224, 224)
    result = main(img)
    for key, value in result.items():
        print(f'result[{key}]:{value}')
