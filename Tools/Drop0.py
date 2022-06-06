import os
from cv2 import imwrite
from datasets.utils.file_utils import readline
from fontTools.afmLib import readlines

label_path = rf'../datasets/coco128_face/labels/train2017'
file1 = os.listdir(str(label_path))

for filename in file1:
    lenLabel = len(readlines(f'{label_path}/{filename}'))
    print(f'{filename}:{lenLabel}')
    listLabel = ''

    for line in open(f'{label_path}/{filename}'):
        print(line)
        if line.split(' ')[0:1][0] != '0':  # TODO label number
            listLabel += line

    path_temp = f'{label_path}/temp/{filename}'

    if not os.path.exists(f'{label_path}/temp'):
        os.makedirs(f'{label_path}/temp')
    with open(path_temp, 'a+') as f:
        f.write(listLabel)

print(f'Bye!')
