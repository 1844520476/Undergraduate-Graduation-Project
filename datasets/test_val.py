"""
input：
1.img_path 数据集图片地址
2.label_path 数据集标签地址（.txt格式）
3.划分比例（0到1） split_rate

output：
--dataset
    --images
        --train
        --val
    --labels
       --train
       --val
"""
import os
import random
import shutil

def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')] #后缀要换成自己的

def move_labels_2(dest_dir, Label_path, Output_path):
    label_list = get_imlist(dest_dir)
    for names in label_list:
        name = names.split('\\')[-1:][0]
        name = name.split('.')[0]
        label = name + '.txt'
        if os.path.exists(rf'{Label_path}/{label}'):
            shutil.copy(rf'{Label_path}/{label}', Output_path)
        else:
            print(f'{label} is not exists')

def move_labels(dest_dir, path):
    label_list = get_imlist(dest_dir + f'/images/{path}')
    for names in label_list:
        name = names.split('\\')[-1:][0]
        name = name.split('.')[0]
        label = name + '.txt'
        if os.path.exists(f'{label_path}/{label}'):
            shutil.copy(f'{label_path}/{label}', dest_dir + f'/labels/{path}')
        else:
            print(f'{label} is not exists')

def main(src_path):
    dest_dir = output_path  # 这个文件夹需要提前建好
    img_list = get_imlist(src_path)
    random.shuffle(img_list)
    length = int(len(img_list) * split_rate)  # 这个可以修改划分比例
    os.makedirs(dest_dir + '/images/train')
    os.makedirs(dest_dir + '/images/test')
    os.makedirs(dest_dir + '/labels/train')
    os.makedirs(dest_dir + '/labels/test')
    for f in img_list[length:]:
        shutil.copy(f, dest_dir + '/images/train')
    for f in img_list[:length]:
        shutil.copy(f, dest_dir + '/images/test')
    #移动对应的标签到对应位置
    move_labels(dest_dir, 'test')
    move_labels(dest_dir, 'train')

if __name__ == '__main__':
    path_dataset = r''
    img_path = r'C:\Users\cleste\Desktop\AutoLabel\datasets\coco128_face\images'
    label_path = r'C:\Users\cleste\Desktop\AutoLabel\datasets\coco128_face\labels'

    split_rate = 0.2

    output_path = r'C:\Users\cleste\Desktop\AutoLabel\datasets\demo'

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    main(img_path)

    # img_path = r'C:\Users\cleste\Desktop\AutoLabel\datasets\DAGM0517_12sp_random\img'
    # label_path = r'C:\Users\cleste\Desktop\machine\machine3\labels'
    # split_rate = 0
    # output_path = r'C:\Users\cleste\Desktop\AutoLabel\datasets\DAGM0517_12sp_random\lab'
    # move_labels_2(img_path, label_path, output_path)

    print(f'finished')