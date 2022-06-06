# -*- coding:utf-8 -*-
from utils.function import *
from PIL import Image
from siamese import Siamese

# 初始化
prob = []
label = []
result = {}
# 实例化网络
model = Siamese()


# 对比（相似度计算）函数
def TwoImg(img, img2):
    try:
        img2 = Image.open(img2)
    except:
        print('Image_2 Open Error! Try again!')

    # 此时img已经转化为PIL格式
    probability = model.detect_image(img, img2)  # 主要函数（孪生网络模型函数）
    # print(probability)
    return probability


# TODO 1.待检测图片地址
def inputImg():
    img = input(f'[1]待检测图片地址:')
    if img == '':
        img = 'query/animals/classcrocodile_(1).jpg'  # （待）检测图片
    print(f'1.待检测图片:{img}')
    return img


# 识别函数
def recognition(Img):
    # 检验待检测图片是否存在
    try:
        img = Image.open(Img)
    except:
        print('Image Open Error! Try again!')

    # TODO 2.标签图片（label images）地址
    img_path = input(f'[2]标签图片：') # 标签图片（label images）
    if img_path == '':
        img_path = rf'support/animals'
    if img_path != 'exit':
        print(f'2.标签图片:{img_path}')

    # 获取标签图片名称等信息
    list = GetFileList(img_path, [])

    # 与每张标签图片进行对比，返回相似度
    for path in list:
        first, last = os.path.splitext(path)
        img2 = path  # 标签图片
        # 对比，输入图片地址，返回相似度（tensor格式）
        prob_two = TwoImg(img, img2)
        # 将相似度tensor格式（prob_two）转化为float格式（similarity）
        similarity = prob_two.tolist()[0]
        # 将相似度结果加入相似度列表（prob）
        prob.append(similarity)
        # 将图片名称加入标签列表（label）
        label.append(first)

    # 打印相似度和图片名称列表
    print(f'\n1.标签为：{label}\n'
          f'2.与待检测图片相似度分别为：{prob}')

    # 对相似度列表进行softmax转换
    prob_sm = softmax(prob)
    print(f'3.各个类别的可能性：{prob_sm}\n')

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


if __name__ == "__main__":
    while True:
        img = inputImg()
        if img == 'exit':
            break
        # img为待检测图片地址
        recognition(img)
