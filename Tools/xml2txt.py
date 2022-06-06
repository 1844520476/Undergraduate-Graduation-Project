import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join


def convert(size, box):
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    x = x_center / size[0]
    y = y_center / size[1]

    w = (box[1] - box[0]) / size[0]
    h = (box[3] - box[2]) / size[1]

    return (x, y, w, h)


def convert_annotation(xml_files_path, save_txt_files_path, classes):
    xml_files = os.listdir(xml_files_path)
    print(xml_files)
    for xml_name in xml_files:
        print(xml_name)
        xml_file = os.path.join(xml_files_path, xml_name)
        out_txt_path = os.path.join(save_txt_files_path, xml_name.split('.')[0] + '.txt')
        out_txt_f = open(out_txt_path, 'w')
        tree = ET.parse(xml_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            # b=(xmin, xmax, ymin, ymax)
            print(w, h, b)
            bb = convert((w, h), b)
            out_txt_f.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


if __name__ == "__main__":
    # 把voc的xml标签文件转化为yolo的txt标签文件
    # 1、类别
    classes1 = ['stop']
    #classes1 = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
    # 2、voc格式的xml标签文件路径
    # xml_files1 = r'home\wyr\桌面\pv_data\Annotations'
    xml_files1 = r'C:\Users\cleste\Desktop\AutoLabel0411\datasets\screen\labels\val_xml'
    # 3、转化为yolo格式的txt标签文件存储路径
    save_txt_files1 = r'C:\Users\cleste\Desktop\AutoLabel0411\datasets\screen\labels\val'
    convert_annotation(xml_files1, save_txt_files1, classes1)
