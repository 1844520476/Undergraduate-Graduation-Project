# 参数类
class opt:
    def __init__(self):
        self.weightspath = None
        self.test_dataloader = None
        self.DatasetName = None
        # label_dict = {'0': 'emotion', '1': 'coco128', '2': 'imagenet', '3': 'cifar10'}
        self.Label_num = 3
        # 每轮测试的图片数量（应该小于test_batch:10000）
        self.epoch = 1
        # 测试轮数
        self.number = 100

    def dataset_name(self):
        print(f'数据集名称是：{self.DatasetName}')