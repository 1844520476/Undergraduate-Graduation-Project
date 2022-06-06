"""
1.先实例化lr,三个参数：learningRate, epoch & alpha，返回LearningRate
2.调用lr_epoch,参数有两个LearningRate和number，返回LearningRate的第number（epoch）的值
"""
import numpy as np
from matplotlib import pyplot as plt


class lr:
    def __init__(self, learning_rate=0.0001, epoch_num_in=1000, alpha=1700):
        self.learning_rate = learning_rate
        self.epoch = epoch_num_in
        self.alpha = alpha
        # self.lrList = self.lossFunction(self.learning_rate, self.epoch_num_in)

    def lossFunction(self):
        # when learningRate plus 100,looks better
        # TODO hypo-parameter:Determine the shape of the curve(max_in between 15~20 is the best)
        max_in = self.learning_rate * self.alpha
        x = np.arange(0, max_in, max_in / float(epoch))
        # 衰减函数，for x:0~epoch, but lr function is lf(>0)~0
        # y = (1/x) * np.sin(x) + \
        #     1 - np.sin(x / max_in * np.pi / 2)
        y = (1 / 100 * (max_in - x)) * np.sin(100 * (max_in - x)) + \
            1 - np.sin(x / max_in * np.pi / 2)
        # - np.sin(x/ max_in * np.pi / 2) 等价于 0 ~ -pi/2的正弦函数曲线走势
        plt.xlabel("x[epoch]")
        plt.ylabel("y[lr]")

        # print contents of function
        plt.title(f"function[alpha={self.alpha}"
                  f"|learningRate={self.learning_rate}"
                  f"|epoch={epoch}]")
        # print(f'original y[{epoch}]:{y}')

        # 后处理
        x_new = x * float(epoch) / max_in
        y_new = abs(y * self.learning_rate)
        # 继续画图
        plt.plot(x_new, y_new)
        plt.show()
        lrList = y_new.tolist()
        return lrList


def lr_epoch(lrList, epochNum):
    lr_in = lrList[epochNum - 1]
    if lr_in < 0.0000001:
        lr_in = 0.0000001
    return lr_in


if __name__ == '__main__':
    i = 0
    while True:
        i += 1
        # alpha input
        alpha_out = input(f'alpha[{i}]:')
        if alpha_out == '':
            alpha_out = 1500000
        # learningRate input
        learningRate = input(f'learning_rate[{i}]:')
        if learningRate == 'exit':
            break
        if learningRate == '':
            learningRate = 0.00001
        # eoich input
        epoch = input(f'epoch_num_in[{i}]:')
        if epoch == '':
            epoch = 100
        # TODO 实例化:注意实参数据类型
        LR = lr(float(learningRate), int(epoch), float(alpha_out))
        # 得到list格式的lr
        LearningRate = LR.lossFunction()
        # print(f'LearningRate:{LearningRate}')
        for number in range(len(LearningRate)):
            # print(i)
            # print(LearningRate[i])
            print(f'LearningRate[{number + 1}]:{LearningRate[number]:.7f}')
        while True:
            while True:
                number = input(f'lr_num[1~{epoch}]:\n')
                if number == 'exit' or number == '':
                    break
                if int(0) < int(number) <= int(epoch):
                    break
            if number == 'exit' or number == '':
                break
            # TODO 得到对应number的lr值
            lr_num = lr_epoch(LearningRate, int(number))
            print(f'\n1. learningRate:{learningRate}\n2. epoch_num_in={epoch}\n3. lr_num[{number}]:{lr_num:.5f}\n')
    print(f'Bye!')
