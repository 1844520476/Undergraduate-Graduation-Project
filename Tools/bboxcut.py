# -*- coding: utf-8 -*-
import os
import cv2

def bboxcut():
    # input path
    inputPath = input('input path:')
    if inputPath == '':
        inputPath = r'datasets/coco128/images/train2017'
    inputPath = str(inputPath)
    print(f'input path:{inputPath}')

    # output  path
    outputPath = input('output path:')
    if outputPath == '':
        outputPath = r'output/bboxcut'
    outputPath = str(outputPath)
    outputPath_label = outputPath + '/labels'
    if not os.path.exists(outputPath_label):
        os.makedirs(outputPath_label)
    print(f'output path:{outputPath}')

    # 输出文件夹根目录
    path = f"{outputPath}"
    # 裁剪出来的小图保存的根目录
    path3 = f"{outputPath}/bboxcut"
    # initialization
    img_total = []
    txt_total = []

    path_img = inputPath
    file1 = os.listdir(str(path_img))
    for filename in file1:
        first, last = os.path.splitext(filename)
        if last == ".jpg":  # 图片的后缀名
            img_total.append(first)

    path_txt = path + '/labels'
    file2 = os.listdir(path_txt)
    for filename in file2:
        first, last = os.path.splitext(filename)
        if last == ".txt":
            txt_total.append(first)

    for txt_ in txt_total:
        if txt_ in img_total:
            filename_img = txt_ + ".txt"  # 图片的后缀名
            pathImg = path_img + f'/{txt_}.jpg'
            img = cv2.imread(pathImg)

            from PIL import Image

            img = Image.open(pathImg)
            w = float(img.width)  # 图片的宽
            h = float(img.height)  # 图片的高
            f = img.format  # 图像格式
            # print(w, h, f)

            filename_txt = txt_ + ".txt"
            n = 1
            with open(os.path.join(path_txt, filename_txt), "r+", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    aa = line.split(" ")
                    x_center = w * float(aa[1])  # aa[1]左上点的x坐标
                    y_center = h * float(aa[2])  # aa[2]左上点的y坐标
                    width = int(w * float(aa[3]))  # aa[3]图片width
                    height = int(h * float(aa[4]))  # aa[4]图片height

                    # 设置裁剪的位置
                    crop_box = (x_center, y_center, width, height)
                    # 裁剪图片
                    img_cut = img.crop(crop_box)
                    # 裁剪出来的小图文件名
                    filename_last = txt_ + f'_class_{aa[0]}_{str(n)}.jpg'
                    #保存图片 path3:f"{outputPath}/bboxcut"
                    import matplotlib.pyplot as plt
                    # 指定图片保存路径
                    figure_save_path = path3
                    if not os.path.exists(figure_save_path):
                        os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
                    # 第一个是指存储路径，第二个是图片名字
                    plt.savefig(os.path.join(figure_save_path, filename_last))
                    n = n + 1

if __name__ == '__main__':
    bboxcut()