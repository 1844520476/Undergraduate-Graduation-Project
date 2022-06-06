# -*- coding:utf-8 -*-
import datetime
from utils.function import *
from PIL import Image
from siamese import Siamese

# 实例化网络
# Model_path = input(f'model weights:')
# if Model_path == '':
#     Model_path = rf'logs/animal300_200/ep039-loss0.038-val_loss0.251.pth'
# print(f'1.模型权重:{Model_path}')
model = Siamese()
# model.generate(Model_path)

# 对比（相似度计算）函数
#相似度函数

def TwoImg(img1, img2, OutPath):
    outputName1 = img1.split('\\')[1].split('.')[0]
    outputName2 = img2.split('\\')[1].split('.')[0]
    outputName = f'{outputName1}_{outputName2}'
    try:
        img = Image.open(img1)
    except:
        print('Image Open Error! Try again!')
    try:
        img2 = Image.open(img2)
    except:
        print('Image_2 Open Error! Try again!')

    # 此时img已经转化为PIL格式
    probability = model.detect_image(img, img2, OutPath, outputName)  # 主要函数（孪生网络模型函数）
    # print(probability)
    return probability

# 识别函数
def recognition(Img, List2, OutPath):
    # 初始化
    prob = []
    label = []
    result = {}
    # 与每张标签图片进行对比，返回相似度
    for Path in List2:
        path = Path.split('\\')[-1]
        first, last = os.path.splitext(path)
        img2 = Path  # 标签图片
        # 对比，输入图片地址，返回相似度（tensor格式）
        prob_two = TwoImg(Img, img2, OutPath)
        # 将相似度tensor格式（prob_two）转化为float格式（similarity）
        similarity = prob_two.tolist()[0]
        # 将相似度结果加入相似度列表（prob）
        prob.append(similarity)
        # 将图片名称加入标签列表（label）
        label.append(first)

    # 打印相似度和图片名称列表
    # print(f'\n1.标签为：{label}\n'
    #       f'2.与待检测图片相似度分别为：{prob}')

    # 对相似度列表进行softmax转换
    prob_sm = softmax(prob)
    # print(f'3.各个类别的可能性：{prob_sm}\n')

    # 打印各个类别的可能性百分比
    for i in range(len(label)):
        print(f'{label[i]}: {prob_sm[i] * 100:.2f}%(相似度：{prob[i] * 100:.5f}%)')
        # 编制result字典：1.key为标签 label[i] 2.value为经softmax的相似度（即概率）prob_sm[i]
        result[f'{label[i]}'] = prob_sm[i]

    # 最有可能的预测结果
    MAX = sorted(result,
                 key=result.get,
                 reverse=True)[0]

    print(f'\n{Img}最有可能的结果是【{MAX}】,'
          f'有{result[MAX] * 100:.2f}%的可能性.\n')
    return prob_sm, result, MAX

#主函数
def main_sn():
    if not input(f'1.only show accuracy:') == 'y':
        # TODO 1.待检测图片地址
        img_path1 = input(f'待检测图片文件夹(query set)地址：')  # 待检测图片文件夹地址
        if img_path1 == '':
            img_path1 = rf'query/mac_test'
        print(f'2.待检测图片:{img_path1}')
        list1 = GetFileList(img_path1, [])

        # TODO 标签图片（label images）地址
        img_path2 = input(f'support set 地址：')
        if img_path2 == '':
            img_path2 = rf'support/mac(DAGM)'
        print(f'3.标签图片:{img_path2}')
        # 获取标签图片名称等信息
        list2 = GetFileList(img_path2, [])

        num = 1
        # 保存文件夹名称
        define = '%Y_%m_%d %H_%M_%S'
        system_time = datetime.datetime.now().strftime(define)
        supportSet = img_path2.split('/')[-1]
        save_path = rf'output/{supportSet}/'
        for img in list1:
            # img为待检测图片地址
            print(f'-----------------------------------------image[{num}]----------------------------------------')
            if not os.path.exists(f'{save_path}/{system_time}/labels'):
                os.makedirs(f'{save_path}/{system_time}/labels')
                OutPath = f'{save_path}/{system_time}'
            prob_sm, result, Max = recognition(img, list2, OutPath)
            with open(rf'{save_path}/{system_time}/info.txt', 'a+') as f:
                f.write(f'{Max} {result}\n')
            first_img, last_img = img.split('\\')
            last_img_excludePNG = last_img.split('.')[:-1][0]
            with open(rf'{save_path}/{system_time}/labels/{last_img_excludePNG}.txt', 'a+') as f:
                f.write(f'{Max}')
            num += 1
        predict_accuracy(rf'{save_path}/{system_time}')
        return 'OK'
    label_path = input(f'labels path：')
    if label_path == '':
        label_path = rf'output/mac(DAGM)/demo'
    predict_accuracy(label_path)

if __name__ == "__main__":
    main_sn()
    while True:
        if input(f'exit?[y/n]:') == 'n':
            main_sn()
        else:
            break
    print('Bye!')