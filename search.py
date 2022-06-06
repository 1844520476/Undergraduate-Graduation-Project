import argparse
import datetime
import math
import sys
import cv2
import os
import time
import CLIP
import torch
from PIL import Image
from pathlib import Path
from dehaze import quwu
from Label.cifar10 import cifar10_label
from Label.custom import *
from Label.emotional import *
from Label.imagenet import *
from Label.coco128 import *
import torch.backends.cudnn as cudnn
from numpy import random
from CustomLabels import class_names_custom, class_numbers_custom, face_names, face_numbers
from Tools.AddLabels import Addlabels
from Tools import TextAdd
from Tools.CudaIsAvailable import CudaIsAvailable
from Tools.OpenLabelWeb import OpenLabelWeb
from Tools.SaveOriImgs import save_oriImgs
from Tools.labels import class_names1, class_numbers1, class_numbers2, class_names2
from lib_forSearch.Utils.plots import imgcut
from lib_forSearch.Models.experimental import attempt_load
from Tools.printTerminal import printTerminal
from lib_forSearch.Utils.datasets import LoadStreams, LoadImages
from lib_forSearch.Utils.general import check_img_size, check_imshow, non_max_suppression, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from lib_forSearch.Utils.plots import plot_one_box
from lib_forSearch.Utils.torch_utils import select_device, time_synchronized

"""
writer: liu ruimeng time；2022.04.19
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple some-package清华镜像
"""


# TODO 只能检测和保存单个yolov5类别的clip细分类

# print Local Time
def LocalTime():
    LocalTime = time.strftime("%Y{y}%m{m}%d{d} %H{h}%M{min}%S{s}").format(y='年', m='月', d='日', h='时', min='分', s='秒')
    print(f'现在是北京时间:{LocalTime}')


global Confidence, alpha


# main function
def search():
    # Local Time
    LocalTime()

    # TODO printTerminal
    printTerminal()

    # 检测目标名字与编号字典
    def choose():
        while True:
            classNum = input(f'请选择label编号【1:coco80、2Mechanical part defect: 、3:custom labels、4：face】：')
            if classNum == '':
                classNum = 1
                classNum = int(classNum)
                break
            elif (classNum == '1') or (classNum == '2') or (classNum == '3') or (classNum == '4'):
                classNum = int(classNum)
                break
            else:
                print('please input 1~3')

        # TODO CustomLabel if you want to detect custom label classes, you need 1.class_names_custom & 2.class_numbers_custom (of 3.course:custom.pt)
        NAMES = (class_names1, class_names2, class_names_custom, face_names)
        NUMBERS = (class_numbers1, class_numbers2, class_numbers_custom, face_numbers)

        classNum = classNum - 1
        class_names = NAMES[classNum]
        class_numbers = NUMBERS[classNum]

        print(f'System path:{sys.path}')
        print('\n以下是阔以被检测的类别^-^These are the number of detection objects:')
        for i in class_names:
            print('class_names[{}] is {}'.format(i, class_names[i]))
        return class_numbers, class_names

    while True:
        class_numbers, class_names = choose()
        if input(f'\nAre you sure:[y/n]\n') == ('y' or ''):
            break

    print('----------------------选一个吧Please choose one-…-搞快点come on------------------------------------')

    # 模型参数选择 TODO weights choose
    Model_path = input('model path[such as : result/P36_v5s6_1to150/weights/best.pt ]:')
    if Model_path == '':
        Model_path = r'pretrained\yolov5s6.pt'  # 文件在项目内部，相对路径就行；如果文件在项目外，则需要绝对地址
    print(rf'pretrained model is :{Model_path}')

    # 筛选用参数
    Confidence = input(f'confidence:[default 0.1]:')  # 检测框的置信度（有多大把握把检测目标正好框进去），大于此的才保留
    if Confidence == '':
        Confidence = 0.1
    Confidence = float(Confidence)
    IOU = input(f'IOU:[default 0.40]:')  # 非极大抑制（IOU代表检测框的重合度），大于此的即舍弃
    if IOU == '':
        IOU = 0.4
    IOU = float(IOU)
    # 相当于模拟退化算法的参数T
    alpha = input(f'Alpha[0.01]:')
    if alpha == '':
        alpha = 0.01
    Img_size = 1280  # 输入图像大小，最好和网络模型对应（640或1280）

    print(f'Confidence:{Confidence},IOU:{IOU},Alpha:{alpha}')

    # # 是否开启二级分类器
    # OpenClassify = False

    while True:
        # 待检测文件地址
        def Input():
            while True:
                Input_path = str(input(rf'under detected file path[default :  datasets/Mac2_100/images/val]:'))
                if Input_path == '':
                    Input_path = f'input/wallpaper'
                    print(f'Input path:{Input_path}')
                return Input_path

        Input_path = Input()

        Aclasses = []  # , 'person', 'car', 'bicycle'
        # TODO 可以直接把想要检测的类别写入Aclasses列表，然后注释掉下面的 Addlabels() 部分。
        #  另外：detect（）只会检测同时包含在Aclasses和classes in model 的类别，所有放心大胆地往Aclasses中塞东西吧
        if input(f'all classes or specific class(es):[all/sp]') == ('sp' or ''):
            Addlabels(class_numbers, class_names, Aclasses)
        else:
            for i in class_names:
                Aclasses.append(class_names[i])
        print(f'classes that you choose:{Aclasses}')

        # 文件夹保存地址
        Save_path = r'output/' + Aclasses[0]

        # 保存文件夹名称
        define = '%Y_%m_%d %H_%M_%S'
        system_time = datetime.datetime.now().strftime(define)
        Save_name = system_time

        # 检测函数
        def detect(save_img=False):

            # 导入opt参数
            source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
            save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
            webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
                ('rtsp://', 'rtmp://', 'http://', 'https://'))
            label_imgs = 0

            # TODO 阈值
            thresholdValue = input(f'是否交由专家标注的阈值[0.50]:')
            if thresholdValue == '':
                thresholdValue = 0.50
            # TODO 底线
            BottomLine = input(f'是否交由专家标注的底线[0.15]:')
            if BottomLine == '':
                BottomLine = 0.15

            # TODO 细粒度分类
            FineGrain = input(f'fineGrain?[y/n]')
            showImg = input(f'show img quickly [T(rue)/F(alse)]')
            if (showImg == '') or (showImg == 'T'):
                showImg = True
            else:
                showImg = False

            if FineGrain == 'y':
                # 模型存储地址
                WeightsPath = r'CustomLabelClassification(ZeroShot)/weights'

                # TODO 选择标签{'0': 'emotion', '1': 'coco128', '2': 'imagenet', '3': 'cifar10', '4': 'custom'}
                label_dict = {'0': 'emotion', '1': 'coco128', '2': 'imagenet', '3': 'cifar10', '4': 'custom'}
                while True:
                    print(f'可选择的【文本】标签：')
                    for i in label_dict:
                        print(f'[{int(i)}]:{label_dict[i]}_label')
                    while True:
                        FineGrainLabel = input(f'Fine Grain Label:')
                        if (FineGrainLabel == '0') or (FineGrainLabel == '1') or (FineGrainLabel == '2') or (
                                FineGrainLabel == '3') or (FineGrainLabel == '4'):
                            break
                    sure = input(f'Are you sure?[y/n]')
                    if sure == 'y' or sure == '':
                        break
                Label_num = int(FineGrainLabel)  # TODO 选择标签，custom_label可以自定义
                label_exist = (emotion_label_clip, coco128_label, imagenet_label, cifar10_label, custom_label_clip)
                Label = label_exist[Label_num]
                print(f'Label:{Label}')

                # 初始化存储概率信息的字典
                dict_prob = {}

                # TODO 对原始标签进行增补操作:
                add1, add2 = TextAdd()

                # 测试并选择设备
                device = 'cuda' if torch.cuda.is_available() else 'cpu'

                # 可选模型
                model_dict = {0: 'RN50', 1: 'RN101', 2: 'RN50x4', 3: 'RN50x16', 4: 'RN50x64',
                              5: 'ViT-B/32', 6: 'ViT-B/16', 7: 'ViT-L/14'}

                # TODO clip模型选择
                model_num = int(5)
                modelClip = model_dict[model_num]

                # TODO 要细分类的标签（FineGrainLabel）选择
                FineGrainClasses = []

                def dict_create(classes_custom):
                    for i in range(len(classes_custom)):
                        CLASS = classes_custom[i]
                        class_numbers_custom[i] = CLASS
                        class_names_custom[CLASS] = i
                    print(f'class_numbers_custom:{class_numbers_custom}\nclass_names_custom:{class_names_custom}')
                    return class_numbers_custom, class_names_custom

                Class_numbers_custom, Class_names_custom = dict_create(Aclasses)

                if input(f'all classes or specific class(es):[all/sp]') == ('sp' or ''):
                    Addlabels(Class_numbers_custom, Class_names_custom, FineGrainClasses)
                else:
                    FineGrainClasses.extend(Aclasses)
                print(f'classes that you choose:{FineGrainClasses}')

                # 加载模型与预处理
                modelClip, preprocess = CLIP.load(modelClip,
                                                  device=device, download_root=WeightsPath)

            # 图像识别函数
            def detectClip(img_path):
                # 确定待配对文本信息
                Label_list = list(Label.values())

                # 提前定义列表
                text_input = []

                for i in range(len(Label_list)):
                    # 将统一修改后的文本添加到新链表中
                    new_label_list = add1 + Label_list[i] + add2
                    text_input.append(new_label_list)

                # 计时器：开始计时
                start_time = time.time()

                # 加载待识别图片
                image = preprocess(Image.open(img_path)). \
                    unsqueeze(0).to(device)

                # 读取文本链表
                text = CLIP.tokenize(text_input).to(device)

                # 开始推理过程
                '''
                torch.no_grad() 是一个上下文管理器，被该语句 wrap 起来的部分将不会track 梯度
                即：被此上下文管理器包裹起来计算部分：可以执行计算，但该计算不会在反向传播中被记录
                '''
                with torch.no_grad():
                    # 提取文本特征信息到向量空间
                    modelClip.encode_image(image)
                    # 提取图像特征信息到向量空间
                    modelClip.encode_text(text)
                    # 计算语义向量与图片特征向量
                    logits_per_image, logits_per_text = modelClip(image, text)
                    # 计算两者的相似度（计算结果以链表形式按顺序存储）
                    probs = logits_per_image.softmax(dim=-1). \
                        cpu().numpy()
                    probs = probs.tolist()
                    # 结束计时
                    Time = time.time() - start_time

                    # 后处理，打印对应概率
                    for i in range(len(text_input)):
                        # 以一维list存储了对应text_input的预测概率
                        prob = probs[0]
                        # 将文本和相似度分别存储为key和value
                        prob[i] = round(prob[i] * 100, 3)
                        dict_prob[Label_list[i]] = prob[i]

                    # 最有可能的预测结果
                    MAX = sorted(dict_prob,
                                 key=dict_prob.get,
                                 reverse=True)[0]

                    probMax = dict_prob[f'{MAX}']

                    result_clip = str(f'{MAX}:{probMax:.2f}%')

                    return result_clip

            # float格式化
            thresholdValue = float(thresholdValue)
            BottomLine = float(BottomLine)

            if BottomLine < opt.conf_thres:
                BottomLine = opt.conf_thres
                print('because BottomLine < opt.conf_thres:')
            print(f'thresholdValue:{thresholdValue},BottomLine:{BottomLine}')

            iter_photo = 0
            # 路径
            save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
            (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
            expertPath = str(save_dir) + fr'\expertLabel{BottomLine}_{thresholdValue}'
            # TODO 必须放在if/else外面，不然出了if/else（即使在函数内部）也是未定义；否则用global定义全局变量

            # 初始化（检测设备硬件和CUDA）
            set_logging()
            device = select_device(opt.device)
            half = device.type != 'cpu'  # half precision only supported on CUDA

            # 加载模型
            model = attempt_load(weights, map_location=device)  # load FP32 model
            stride = int(model.stride.max())  # model stride
            imgsz = check_img_size(imgsz, s=stride)  # check img_size
            if half:
                model.half()  # to FP16

            # # 二级分类器
            # classify = OpenClassify
            # if classify:
            #     modelc = load_classifier(name='resnet101', n=2)  # 初始化
            #     modelc.load_state_dict(
            #         torch.load(r'C:\Users\cleste/.cache\torch\hub\checkpoints\resnet101-5d3b4d8f.pth', map_location=device)[
            #             'model']).to(device).eval()

            # 设置Dataloader
            vid_path, vid_writer = None, None
            if webcam:
                view_img = check_imshow()
                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadStreams(source, img_size=imgsz, stride=stride)
            else:
                dataset = LoadImages(source, img_size=imgsz, stride=stride)

            # 获取名称和颜色
            names = model.module.names if hasattr(model, 'module') else model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

            # 运行inference
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
            t0 = time.time()
            for path, img, im0s, vid_cap in dataset:
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                t1 = time_synchronized()
                pred = model(img, augment=opt.augment)[0]

                # 应用NMS
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                           agnostic=opt.agnostic_nms)
                t2 = time_synchronized()

                # # 应用分类器
                # if classify:
                #     pred = apply_classifier(pred, modelc, img, im0s)

                # TODO 检测过程（循环存储照片的文件夹）
                for i, det in enumerate(pred):
                    # detections per image
                    if webcam:  # batch_size >= 1
                        p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                    else:
                        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                    p = Path(p)  # to Path
                    save_path = str(save_dir / p.name)  # img.jpg
                    txt_path = str(save_dir / 'labels' / p.stem) + (
                        '' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                    s += '%gx%g ' % img.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    if len(det):
                        # 将检测框从 img_size 缩放到 im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # print 结果:输出信息：检测到的各个类别与个数
                        for c in det[:, -1].unique():
                            if names[int(c)] in Aclasses:  # 只检测Aclasses这特定的类
                                n = (det[:, -1] == c).sum()  # 检测每个类
                                s += f'\n-----------------------------------------------------------这是第{label_imgs + 1}个检测目标-----------------------------------------------------------'
                                s += f"\n检查结果为   类别：{names[int(c)]}，其数量：{n} 。{'不止一个哦！' * (n > 1)} "  # 添加检测结果说明
                                label_imgs += 1

                        # Write 结果:
                        # 1.保存预测信息: txt
                        # 2.在图像上画框
                        # 3.crop_img（裁切图片）
                        for *xyxy, conf, cls in reversed(det):
                            # TODO 对于在Aclass中的类别
                            if names[int(cls)] in Aclasses:
                                if save_img or view_img:  # Add bbox to image

                                    if FineGrain == 'y':  # 要细分类
                                        saveFilePath = f'{Save_path}/{Save_name}'
                                        # TODO 对于在Aclass中的类别：二次（细）分类与否
                                        # FineGrainClasses = input(f'细分类类别:')
                                        # if FineGrainClasses == '':
                                        #     FineGrainClasses = f'person'
                                        # TODO 图片检测框（FineGrainClassification）细分类
                                        if names[int(cls)] in FineGrainClasses:  # 要细分类的（要检测标签的子集）标签
                                            # 待细分类图片地址
                                            # TODO 裁切图片并存储
                                            photopath, clipnmae = imgcut(path, xyxy, saveFilePath,
                                                                         p.name)  # img-cut函数在plots.py模块中
                                            # 防止对应地址无细分类检测图片的情况
                                            if os.path.exists(photopath):
                                                # TODO clip（ZeroShot）自定义标签细分类
                                                Result_clip = detectClip(photopath)  # clip细（二次）分类函数
                                            else:
                                                Result_clip = ''

                                            # TODO better to use conf(idence) & 在图片上绘制标签
                                            label = f'{names[int(cls)]}:{conf * 100:.2f}%' \
                                                    + f' {Result_clip}'  # 细分类结果：类别+概率
                                            print(f'{photopath} {Result_clip}')
                                            # 将检测数据统一写入包含细粒度识别的txt文件
                                            with open(str(save_dir) + rf'\clip\FineGrain.txt', 'a') as f:
                                                f.write(f'{clipnmae} ' + names[
                                                    int(cls)] + f' {Result_clip}\n')  # 将二次分类结果写入clip\FineGrain.txt文本文件
                                        else:  # 无需细分类的（要检测标签的子集）标签
                                            label = f'{names[int(cls)]}: {conf * 100:.2f}%'
                                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                                    else:  # 无需细分类，仅目标检测
                                        label = f'{names[int(cls)]}: {conf * 100:.2f}%'
                                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                                if save_txt:  # Write to file
                                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                        -1).tolist()  # normalized xywh
                                    line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label 格式化

                                    # TODO 单独保存conf从阈值(thresholdValue)到1之间 & conf到底线(BottomLine)的标签
                                    # 预先准备好相应的文件
                                    expertLabelPath = expertPath + fr'\labels'
                                    # 不存在则创建
                                    if not os.path.exists(expertLabelPath):
                                        os.makedirs(expertLabelPath)

                                    if (float(conf) <= BottomLine) or (float(conf) > thresholdValue):
                                        # TODO 产生0到1间的随机数和模拟退化概率公式
                                        p_SA = math.exp(-(float(conf) - thresholdValue) / (1 - thresholdValue) / alpha)
                                        if 1 > random.random() > p_SA:
                                            with open(txt_path + '.txt', 'a') as f:
                                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                                        else:
                                            with open(expertLabelPath + rf'\{p.stem}.txt', 'a') as f:
                                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                                    # elif float(conf) <= BottomLine:
                                    #     expertLabelPath0 = str(save_dir / f'{Confidence}_{BottomLine}/labels')
                                    #     if not os.path.exists(expertLabelPath0):
                                    #         os.makedirs(expertLabelPath0)
                                    #     txt_path2 = str(expertLabelPath0 / p.stem) + (
                                    #         '' if dataset.mode == 'image' else f'_{frame}')
                                    #     with open(txt_path2 + '.txt', 'a') as f:
                                    #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

                                    # TODO 单独保存conf从底线到阈值之间的标签
                                    else:
                                        with open(expertLabelPath + rf'\{p.stem}.txt', 'a') as f:
                                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                                    # 将检测数据统一写入包含置信度的txt文件
                                    with open(str(save_dir) + r'\confidence.txt', 'a') as f:
                                        f.write(f'{p.stem} ' + ('%g ' * len(line)).rstrip() % line + f' {conf}\n')

                    # Print time (inference + NMS)
                    print(f'{s}')
                    print(f'这张图片搞定了！inference 与NMS共耗时：{t2 - t1:.3f}秒\n')

                    # Stream results:是否需要显示预测后的结果  img0(此时已将pred结果可视化到了img0中)

                    # Save results (image with detections):是否需要保存图片或视频（检测后的图片/视频 里面已经被画好了框的） img0
                    if save_img:
                        if names[int(cls)] in Aclasses:
                            cv2.imwrite(save_path, im0)
                            # 保存原图
                            save_im0s = 1
                            if save_im0s:
                                cv2.imwrite(save_path, im0s)
                    else:  # 'video' or 'stream'
                        if names[int(cls)] in Aclasses:
                            if vid_path != save_path:  # new video
                                vid_path = save_path
                                if isinstance(vid_writer, cv2.VideoWriter):
                                    vid_writer.release()  # release previous video writer
                                if vid_cap:  # video
                                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                else:  # stream
                                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                                    save_path += '.mp4'
                                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                            vid_writer.write(im0)

                    # TODO --view-img
                    if view_img:
                        if names[int(cls)] in Aclasses:
                            if showImg:
                                cv2.imshow(str(p), im0)
                                cv2.waitKey(2)  # 图片显示停留时间
                            else:
                                import matplotlib
                                matplotlib.use('TkAgg')
                                import matplotlib.pyplot as plt
                                if iter_photo != 0:
                                    # 关闭幕布
                                    plt.close()
                                iter_photo += 1
                                # opencv的颜色通道顺序为[B,G,R]，而matplotlib颜色通道顺序为[R,G,B],所以需要调换一下通道位置
                                img1 = cv2.imread(str(p))[:, :, (2, 1, 0)]
                                img2 = cv2.imread(save_path)[:, :, (2, 1, 0)]
                                # 结果展示
                                plt.subplot(121)
                                plt.imshow(img1)  # imshow()对图像进行处理，画出图像，show()进行图像显示
                                plt.title(f'[{iter_photo}]Original image')
                                plt.axis('on')  # 显示坐标轴
                                # 处理后的图片
                                plt.subplot(122)
                                plt.imshow(img2)
                                plt.title(f'[{iter_photo}]Processed images')
                                plt.axis('on')
                                # subplot parameters
                                plt.subplots_adjust(left=0.04, bottom=0.1, right=0.99, top=0.97, wspace=0.11,
                                                    hspace=0.45)
                                # 设置子图默认的间距
                                plt.tight_layout()
                                # 显示图像
                                plt.show()
                                plt.pause(1)

            if not showImg:
                plt.close()
            if save_txt or save_img:
                s = f"\n{len(list(save_dir.glob('labels/*.txt')))} 个标签已经保存到了以下地址： {save_dir / 'labels'}" if save_txt else ''
                print(f"{s}")

            print(f'检测程序结束运行，总共耗时：{time.time() - t0:.3f}秒')
            return expertPath

        # parameters configuration
        parser = argparse.ArgumentParser()
        # 模型参数（pt文件）地址
        parser.add_argument('--weights', nargs='+', type=str, default=Model_path, help='model.pt path(s)')
        # 待检测目标（会读取文件夹下所有）
        parser.add_argument('--source', type=str, default=Input_path, help='source')  # file/folder, 0 for webcam
        # 输入图像大小，最好和网络模型对应（640或1280）
        parser.add_argument('--img-size', type=int, default=Img_size, help='inference size (pixels)')
        # 检测框的置信度（有多大把握把人物正好框进去）
        parser.add_argument('--conf-thres', type=float, default=Confidence, help='object confidence threshold')
        # 非极大抑制（IOU代表检测框的重合度）
        parser.add_argument('--iou-thres', type=float, default=IOU, help='IOU threshold for NMS')
        # 硬件选择（CPU还是GPU）
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        # 以下需可以在  编辑配置  中添加  形参  以决定False or True（默认为False，输入--view-img这一项便为True：view-img = True）
        # 是否显示检测过程（img by img）
        parser.add_argument('--view-img', action='store_true', help='display results')
        # 是否在保存labels文件.txt
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        # 是否保存置信度
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        # 是否保存被检测对象
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        # 是否增强NMS，会更加强大、提升结果
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        # 同是增强检测效果
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        # 为True只保留本模型需要的东西
        parser.add_argument('--update', action='store_true', help='update all Models')
        # 保存文件同名时是否新增一个文件夹保存（为True会保存在同名的文件夹下）
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        # 决定保留class多少的结果（--class 0）
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        # 保存地址
        parser.add_argument('--project', default=Save_path, help='save results to project/name')
        # 保存文件名
        parser.add_argument('--name', default=Save_name, help='save results to project/name')

        opt = parser.parse_args()
        print(f'parser.parse_args(分析程序的参数):\n{opt}')
        # check_requirements(exclude=('pycocotools', 'thop'))

        with torch.no_grad():
            if opt.update:  # 更新所有模型(修复SourceChangeWarning)
                for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                    ExpertPath = detect()
                    strip_optimizer(opt.weights)
            else:
                ExpertPath = detect()

        print(
            f'\nHi,congratulation!Don t forget '
            f'\n1.pt file path:{Model_path}'
            f'\n2.imgs path:{Input_path}'
            f'\nNow,you can see the result in {Save_path}/{Save_name}'
            f'\nAnd then you can fix these labels in labelling or webs such as makesence! '
            f'Because human brain is the best computer for now,lol ^_^')

        if input(f'\nDo you want to save original photos with their labels in save directory?[y/n]:') == 'y':
            # TODO: save original photos with their labels
            Save_path = Save_path + '/' + Save_name
            print(f'以下是直接参与后续微调的图片:')
            save_oriImgs(Save_path, Input_path)
            print(f'以下是expertLabel images:')
            save_oriImgs(ExpertPath, Input_path)
        else:
            print(
                f'\nHi hi hi, therefore you don t want to save original  photos, I have to remind you again: \nORIGINAL PHOTOS PATH:{Input_path}')

        if input(f'\nAre you sure to reconfiguration[y/n]:') == 'y':
            break

    return Save_path


if __name__ == '__main__':
    # global ExpertPath

    # cuda可用性检查
    CudaIsAvailable()

    # main recurrent
    while True:
        datasetPath = search()

        print('\nIf you don t want to exit please [input n], or you can reconfiguration:')
        if input(f'Are you sure to exit:') == 'exit':
            break

    # 打开标注网站（以便人工修改）
    Relabelled = input(f'\nHi hi,I forget one thing ! Do you want to Manual finishing labeling now?[y/n]:')
    if Relabelled == 'y':
        # labelling Web introduction blog: https://blog.csdn.net/To_ChaRiver/article/details/119619515
        OpenLabelWeb()

    print(f'See you next time and dont forget you dataset are sleeping in :{datasetPath} !')
