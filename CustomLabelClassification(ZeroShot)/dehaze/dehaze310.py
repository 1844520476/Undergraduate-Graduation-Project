"""
1.去雾算法介绍博客：
http://blkstone.github.io/2015/08/20/single-image-haze-removal-using-dark-channel/
2.论文地址（2009 CVPR best paper）：
https://paperswithcode.com/paper/single-image-haze-removal-using-dark-channel
3.资源下载（镜像）地址：
pip install [The Package You Want to Download] -i https://pypi.tuna.tsinghua.edu.cn/simple
ps. cv2 means opencv-python
"""
import time

import cv2
import numpy as np
import argparse


# 计算雾化图像的暗通道
def DarkChannel(img, size=15):
    """
    暗通道的计算主要分成两个步骤:
    1.获取BGR三个通道的最小值
    2.以一个窗口做MinFilter
    ps.这里窗口大小一般为15（radius为7）
    获取BGR三个通道的最小值就是遍历整个图像，取最小值即可
    """
    r, g, b = cv2.split(img)
    min_img = cv2.min(r, cv2.min(g, b))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dc_img = cv2.erode(min_img, kernel)
    return dc_img


# 估算全局大气光值
def GetAtmo(img, percent=0.001):
    """
    1.计算有雾图像的暗通道
    2.用一个Node的结构记录暗通道图像每个像素的位置和大小，放入list中
    3.对list进行降序排序
    4.按暗通道亮度前0.1%(用percent参数指定百分比)的位置，在原始有雾图像中查找最大光强值
    """
    mean_perpix = np.mean(img, axis=2).reshape(-1)
    mean_topper = mean_perpix[:int(img.shape[0] * img.shape[1] * percent)]
    return np.mean(mean_topper)


# 估算透射率图
def GetTrans(img, atom, w):
    """
    w为去雾程度，一般取0.95
    w的值越小，去雾效果越不明显
    """
    x = img / atom
    t = 1 - w * DarkChannel(x, 15)
    return t


def GuidedFilter(p, i, r, e):
    """
    :param p: input image
    :param i: guidance image
    :param r: radius
    :param e: regularization
    :return: filtering output q
    """
    # 1
    mean_I = cv2.boxFilter(i, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    corr_I = cv2.boxFilter(i * i, cv2.CV_64F, (r, r))
    corr_Ip = cv2.boxFilter(i * p, cv2.CV_64F, (r, r))
    # 2
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    # 3
    a = cov_Ip / (var_I + e)
    b = mean_p - a * mean_I
    # 4
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    # 5
    q = mean_a * i + mean_b
    return q


# 去雾主程序
def DeHaze(opt):
    path, output, photo, t0, w = opt.input, opt.output, opt.photo, opt.threshold_value, opt.dehaze_degree
    # 读取待处理图像
    im = cv2.imread(path)
    # 压缩RGB通道值于0到1
    img = im.astype('float64') / 255
    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype('float64') / 255

    atom = GetAtmo(img)
    trans = GetTrans(img, atom, w)
    trans_guided = GuidedFilter(trans, img_gray, 20, 0.0001)
    """
    1.t0 最小透射率值，一般取0.25
    2.投射图t 的值过小——>图像会整体向白场过度
    3.因此一般设置一阈值t0：当t值小于t0时，令t=t0
    """
    trans_guided = cv2.max(trans_guided, t0)

    result = np.empty_like(img)
    for i in range(3):
        result[:, :, i] = (img[:, :, i] - atom) / trans_guided + atom

    # 显示&保存结果
    cv2.imshow("source", img)
    cv2.imshow("result", result)
    cv2.waitKey()

    photoName = r'{}\{}'.format(output, photo)

    if output is not None:
        # TODO 重名照片问题处理 现已解决:在名字后添加保存时间 %d%H%M(日、时、分)
        if photoName is not None:
            photo = photo.split(".")
            photo_temp = str(photo[0] + f'_{time.strftime("%m%d%H%M", time.localtime())}')
            photo = str(photo_temp + '.' + photo[1])
            cv2.imwrite("{}\\{}".format(output, photo), result * 255)
        else:
            cv2.imwrite("{}\\{}".format(output, photo), result * 255)

    return photo_temp


# # 可通过命令行传递参数
# parser = argparse.ArgumentParser()
# parser.add_argument('-i', '--input', default=ImgInput)
# parser.add_argument('-o', '--output', default=ImgFile)
# parser.add_argument('-p', '--photo', default=photo_name)
# # t为最小透射率，一般取0.25。投射图t 的值过小——>图像会整体向白场过度
# parser.add_argument('-t', '--threshold_value', default=0.50)
# # w为去雾程度，一般取0.95。w的值越大，去雾效果越明显
# parser.add_argument('-w', '--dehaze_degree', default=0.20)
# opt = parser.parse_args()
# print(f'parser.parse_args(解析器的参数):\n{opt}')


class opt:
    def __init__(self):
        self.input = None
        self.output = None
        self.photo = None
        self.threshold_value = 0.25
        self.dehaze_degree = 0.90

    def opt(self, ImgInput, ImgFile, photo_name):
        self.input = ImgInput
        self.output = ImgFile
        self.photo = photo_name


# 主程序
def quwu(img):
    img_ed = None
    # 图片相关参数：
    # 1.默认图片
    Defeat = r'canon3.bmp'
    defeat_photo = '..\\Input\\haze\\{}'.format(Defeat)

    if img is None:
        # 待处理图片
        photo_name = f'{Defeat}.jpg'
        # 处理后图片保存地址
        imgFile = 'Output\\HazeRemove'

        ImgInput = defeat_photo
        ImgFile = r'..\{}'.format(imgFile)

    else:
        # 2.待处理图片
        photo_name = f'{img}.jpg'
        # 3.待处理图片所在目录地址
        imgInput = 'Input\\emotion'
        # 4.处理后图片保存地址
        imgFile = r'Output/haze_test'

        ImgInput = r'..\{}\{}'.format(imgInput, photo_name)
        ImgFile = r'..\{}'.format(imgFile)

    Opt = opt()
    Opt.opt(ImgInput, ImgFile, photo_name)

    if __name__ == '__main__':  # 这里必须是__main__（在本模块执行的时候__name__会保存为__main__），不能是__DeHaze__或__dehaze__(函数/模块名)
        img_ed = DeHaze(Opt)
        print('[1]去雾算法正在dehaze模块中被测试')
    else:
        img_ed = DeHaze(Opt)
        print('[2]去雾算法正在被调用')
    return img_ed, imgFile


for i in range(7):
    i += 1
    img_trans = 'haze' + str(i)
    ImgFile, ImgEd = quwu(img_trans)
    print('存储的新图片地址{}和名称{}'.format(ImgFile, ImgEd))

# img_trans = None
# ImgFile, ImgEd = quwu(img_trans)
