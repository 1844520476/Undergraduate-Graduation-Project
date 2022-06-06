import os
import datetime

import numpy as np
import os.path


# 输入文件夹地址和[]，返回包含子文件名称的列表
def GetFileList(dir, fileList):
    newDir = dir
    if os.path.isfile(dir):
        fileList.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            # 如果需要忽略某些文件夹，使用以下代码
            # if s == "xxx":
            # continue
            newDir = os.path.join(dir, s)
            GetFileList(newDir, fileList)
    return fileList


# softmax转换 输入输出为列表
def softmax(x):
    """Compute the softmax in a numerically stable way."""
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x

def predict_accuracy(result_path):
    fileList = []
    result_path = rf'{result_path}\labels'
    filelist = GetFileList(result_path, fileList)
    total = len(filelist) - 1
    num = 0
    acc = 0
    for txt in filelist:
        txtPath = txt.split('\\')[-1:][0]
        if txtPath != 'predict_accuracy.txt':
            class_num, number_classes = txtPath.split('_')
            if not os.path.exists(txt):
                print(f'txt文件地址不存在')
            with open(txt, 'r') as f:
                predict_result = f.readline()
            if predict_result == class_num[5:]:
                acc += 1
            num += 1
            print(f'[{num}]预测准确率：{float(acc/num)*100:.2f}%\n进度：{float(num/total)*100:.2f}%')
    with open(rf'{result_path}/predict_accuracy.txt', 'a+') as f:
        predict_acc = f'{float(acc/num)*100:.2f}'
        define = '%Y_%m_%d %H_%M_%S'
        system_time = datetime.datetime.now().strftime(define)
        Time = system_time
        f.write(f'system_time: {Time}\npredict_accuracy: {predict_acc} %\n')

def demo(path):
    list = []
    for line in open(path):
        line_first = str(line.split('\\')[1])
        print(line_first)
        list.append(line_first)
    f = open(path,'w')
    for i in range(len(list)):
        f.write(list[i])


if __name__ == '__main__':
    # 测试GetFileList函数
    list = GetFileList('../support/text', [])
    for path in list:
        print(path)
    predict_accuracy(rf'../output/mac(DAGM)/2022_05_01 01_51_04')

    #demo(r'C:\Users\cleste\Desktop\孪生网络（少样本分类）\demo.txt')

