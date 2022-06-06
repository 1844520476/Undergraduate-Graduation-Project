import shutil  # 这个库复制文件比较省事
import os
from PIL import Image

Input_path = r'input/RealPhonePhotos'
Save_path = r'output/person/2022_04_17 22_25_21'

def save_oriImgs(save_path, input_path):
    #TODO 默认子文件夹分为labels&images
    # label_path = save_path + r'/labels/' （读取文件名地址）
    # save_path = save_path + r'/images' （保存文件夹地址）
    label_path = save_path + r'/labels/'
    imgs_labels = []
    file1 = os.listdir(str(label_path))
    file2 = os.listdir(str(input_path))
    for filename in file1:
        first, last = os.path.splitext(filename)
        if last == ".txt":  # 图片的后缀名
            imgs_labels.append(first)

    save_path = save_path + '/images'
    # 不存在则创建
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    dirname_read = rf"{input_path}/"  # 注意后面的斜杠
    dirname_write = rf"{save_path}/"

    for i in range(0, len(imgs_labels)):
        for filename in file2:
            first2, last2 = os.path.splitext(filename)
            if first2 == str(imgs_labels[i]):
                imgs_labels[i] = first2 + last2

    print(imgs_labels)
    print(len(imgs_labels))
    print(dirname_read)
    for i in imgs_labels:
        new_obj_name = i
        if new_obj_name in imgs_labels:
            print(new_obj_name)
            shutil.copy(dirname_read + '/' + new_obj_name, dirname_write + '/' + new_obj_name)

if __name__ == '__main__':
    save_oriImgs(Save_path, Input_path)
