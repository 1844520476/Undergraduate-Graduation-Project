#TODO 增加能生成类别标签文件的程序

def crelab():
    # 数据集名称
    name = 'dataset Name'
    # 标签类别名称
    names = [
        'Label Name'
    ]

    # 初始化标签字典
    cifar100 = {}

    # 计算标签个数
    length = int(len(names))
    print(f'{name}的类别数为：{length}')

    # 写入标签名字到字典
    for i in range(length):
        cifar100[i] = names[i]
    print(f'{name}:{cifar100}')

#label_exist = (emotion_label, coco128_label, imagenet_label, cifar10_label)
#label_dict = {'0': 'emotion', '1': 'coco128', '2': 'imagenet', '3': 'cifar10'}
