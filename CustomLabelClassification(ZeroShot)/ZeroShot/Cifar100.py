"""
1.CLIP源代码：https://github.com/OpenAI/CLIP.
2.请在终端中运行下面两行代码（如果是在ipynb环境下运行：请在pip前加上！）
    pip install ftfy regex tqdm
    pip install git+https://github.com/openai/CLIP.git
3.国内镜像源：(1)清华镜像:pip install [The Package You Want to Download]
-i https://pypi.tuna.tsinghua.edu.cn/simple
4.刘同学于2022.03.11增补
"""

# 导入所需组件
import torchvision.datasets
from torch.utils.data import DataLoader
from Option import opt
from ZeroShot import main_zeroshot


# 数据加载函数
def data_in(Opt):
    # 数据集准备
    test_data = torchvision.datasets.CIFAR100(r"..\Dataset\Cifar100", train=False,
                                              transform=torchvision.transforms.ToTensor(), download=True)

    # length长度
    test_data_size = len(test_data)
    print("测试数据集的长度为{}".format(test_data_size))

    # 利用dataloader加载数据集
    test_dataloader = DataLoader(test_data, batch_size=1)
    print(f'数据已加载完成')

    # 确定epoch与number
    Opt.number = int(input(f'请输入number[检测图片数量]：'))
    Opt.epoch = int(input(f'请输入epoch[检测轮数]：'))
    # 导入数据集名称
    Opt.DatasetName = 'CIFAR100'
    # 对应标签选择
    '''
    #待测数据集标签选择
    label_exist = (emotion_label, coco128_label, imagenet_label, cifar10_label)
    label_dict = {'0': 'emotion', '1': 'coco128', '2': 'imagenet', '3': 'cifar10'}
    要添加对应的数据集标签（及中文翻译）,此文件就是添加cifar100_label
    '''
    Opt.Label_num = 4
    # 模型存储地址
    Opt.weightspath = r'..\weights'
    # 数据集加载
    Opt.test_dataloader = test_dataloader
    return test_dataloader, Opt

def start():
        while True:
            # 初始化参数类
            Opt = opt()
            # 加载数据与参数
            test_dataloader, Option = data_in(Opt)
            # accuracy = main_zeroshot(Option)
            # print(f'accuracy:{accuracy}')
            # 退出与否
            exit = input('是否退出系统:[y/n]')
            if exit == 'y':
                print(f'-----------------------------期待您再次使用系统，再见>_<！--------------------------------')
                break
            else:
                print('即将返回主界面')

# 执行主程序
if __name__ == '__main__':
    start()