import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torch import nn

from nets.siamese import Siamese as siamese

#中文字体参数设置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'FangSong']  # 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
plt.rcParams['font.size'] = 10  # 字体大小
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# ---------------------------------------------------#
#   使用自己训练好的模型预测需要修改model_path参数
# ---------------------------------------------------#
class Siamese(object):
    _defaults = {
        # -----------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path
        #   model_path指向logs文件夹下的权值文件
        # -----------------------------------------------------#
        # -----------------------------------------------------#
        #   输入图片的大小。
        # -----------------------------------------------------#
        # TODO  2.（先）高 （后）宽 RGB三通道
        "input_shape": (512, 512, 3),
        # -------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # -------------------------------#
        "cuda": True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化Siamese
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        # TODO 1.（预测阶段）选择网络权重，记得要让input_shape与weights.pt对应
        ModelPath = input(f'[0]model path:')
        if ModelPath != '':
            self.model_path = ModelPath
        else:
            self.model_path = 'logs/mac_10/ep121-loss0.000-val_loss0.000.pth'
        print(f'model path: %s' % self.model_path)
        self.generate()
        print(f'(predict)input_shape:{self.input_shape}')

    # ---------------------------------------------------#
    #   载入模型
    # ---------------------------------------------------#
    def generate(self):
        # ---------------------------#
        #   载入模型与权值
        # ---------------------------#
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = siamese(self.input_shape)
        model.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = model.eval()
        print('{} model loaded.'.format(self.model_path))

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()

    def letterbox_image(self, image, size):
        image = image.convert("RGB")
        iw, ih = image.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        if self.input_shape[-1] == 1:
            new_image = new_image.convert("L")
        return new_image

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image_1, image_2, save_path, output_name):
        # ---------------------------------------------------#
        #   对输入图像进行不失真的resize
        # ---------------------------------------------------#
        image_1 = self.letterbox_image(image_1, [self.input_shape[1], self.input_shape[0]])
        image_2 = self.letterbox_image(image_2, [self.input_shape[1], self.input_shape[0]])

        # ---------------------------------------------------#
        #   对输入图像进行归一化
        # ---------------------------------------------------#
        photo_1 = np.asarray(image_1).astype(np.float64) / 255
        photo_2 = np.asarray(image_2).astype(np.float64) / 255

        if self.input_shape[-1] == 1:
            photo_1 = np.expand_dims(photo_1, -1)
            photo_2 = np.expand_dims(photo_2, -1)

        with torch.no_grad():
            # ---------------------------------------------------#
            #   添加上batch维度，才可以放入网络中预测
            # ---------------------------------------------------#
            photo_1 = torch.from_numpy(np.expand_dims(np.transpose(photo_1, (2, 0, 1)), 0)).type(torch.FloatTensor)
            photo_2 = torch.from_numpy(np.expand_dims(np.transpose(photo_2, (2, 0, 1)), 0)).type(torch.FloatTensor)

            if self.cuda:
                photo_1 = photo_1.cuda()
                photo_2 = photo_2.cuda()

            # ---------------------------------------------------#
            #   获得预测结果，output输出为概率
            # ---------------------------------------------------#
            output = self.net([photo_1, photo_2])[0]
            output = torch.nn.Sigmoid()(output)

        if float(output) > 0: # 置信度大于1%的图片对才会被打印
            plt.subplot(1, 2, 1)
            plt.imshow(np.array(image_1))

            plt.subplot(1, 2, 2)
            plt.imshow(np.array(image_2))
            similarity = f'相似度:{float(output)*100:.2f}%'
            plt.text(-12, -12, similarity, ha='center', va='bottom', fontsize=11)
            #plt.text(-12, -12, 'Similarity:%.3f' % output, ha='center', va='bottom', fontsize=11)

            f = plt.gcf()
            f.savefig(f'{save_path}/{output_name}')

            plt.show()
        return output


