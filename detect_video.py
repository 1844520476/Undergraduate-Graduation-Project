import argparse
import os
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional
from numpy import random
from PIL import Image
from torch.nn.modules import transformer
from torchvision import transforms
import CLIP
from CustomLabels import class_numbers_custom, class_names_custom
from Label.cifar10 import cifar10_label
from Label.coco128 import coco128_label
from Label.custom import custom_label_clip
from Label.emotional import emotion_label_clip, emotion_label_clip2
from Label.imagenet import imagenet_label
from Tools import TextAdd
from Tools.AddLabels import Addlabels
from lib_forSearch.Models.experimental import attempt_load
from lib_forSearch.Utils.datasets import LoadStreams, LoadImages
from lib_forSearch.Utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
    apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from lib_forSearch.Utils.plots import plot_one_box, imgcut
from lib_forSearch.Utils.torch_utils import select_device, load_classifier, time_synchronized


# TODO 一定要记得输入形参：--save-txt --view-img

def detect(save_img=False, vutils=None):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images

    # TODO 1.细粒度分类
    # 模型存储地址
    WeightsPath = r'CustomLabelClassification(ZeroShot)/weights'
    label_dict = {'0': 'emotion', '1': 'coco128', '2': 'imagenet', '3': 'cifar10', '4': 'custom'}
    # while True:
    #     print(f'可选择的【文本】标签：')
    #     for i in label_dict:
    #         print(f'[{int(i)}]:{label_dict[i]}_label')
    #     while True:
    #         FineGrainLabel = input(f'Fine Grain Label:')
    #         if (FineGrainLabel == '0') or (FineGrainLabel == '1') or (FineGrainLabel == '2') or (
    #                 FineGrainLabel == '3') or (FineGrainLabel == '4'):
    #             break
    #     sure = input(f'Are you sure?[y/n]')
    #     if sure == 'y' or sure == '':
    #         break
    FineGrainLabel = '0'
    Label_num = int(FineGrainLabel)
    label_exist = (emotion_label_clip2, coco128_label, imagenet_label, cifar10_label, custom_label_clip)
    Label = label_exist[Label_num]
    print(f'Label:{Label}')
    # 初始化存储概率信息的字典
    dict_prob = {}
    # 对原始标签进行增补操作
    add1, add2 = TextAdd()
    # 测试并选择设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 模型选择
    modelClip = 'ViT-B/32'

    # TODO 1.1要细分类的标签（FineGrainLabel）1.face 2. person
    FineGrainClasses = ['face']

    print(f'classes that you choose:{FineGrainClasses}')

    # 加载模型与预处理
    modelClip, preprocess = CLIP.load(modelClip,
                                      device=device, download_root=WeightsPath)

    # 图像识别函数
    def detectClip(img_clip):
        img_path = torchvision.transforms.functional.to_pil_image(img_clip)
        # 确定待配对文本信息
        Label_list = list(Label.values())
        # 提前定义列表
        text_input = []
        for i in range(len(Label_list)):
            # 将统一修改后的文本添加到新链表中
            new_label_list = add1 + Label_list[i] + add2
            text_input.append(new_label_list)
        # 加载待识别图片
        image = preprocess(img_path). \
            unsqueeze(0).to(device)
        # 读取文本链表
        text = CLIP.tokenize(text_input).to(device)
        # 开始推理过程
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

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    num = 1
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        if names[int(cls)] in FineGrainClasses:
                            # TODO 2.原图与缩略图尺寸
                            # x1, y2, x3, y4 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                            # x_ratio = im0.shape[0] / img.shape[2]
                            # y_ratio = im0.shape[1] / img.shape[3]
                            # c1, c2, c3, c4 = int(x_ratio * x1), int(x_ratio * x3 + 1),int(y_ratio * y2), int(y_ratio * y4 + 1)
                            height = im0.shape[0]
                            width = im0.shape[1]
                            c1, c2, c3, c4 = int(width * (line[1] - line[3] / 2)), int(width * (line[1] + line[3] / 2)), \
                                             int(height * (line[2] - line[4] / 2)), int(
                                height * (line[2] + line[4] / 2))

                            img_ndarray = im0[c3:c4, c1:c2, :]
                            # ndarray转化为tensor
                            img_tensor = torch.from_numpy(img_ndarray) / 255
                            img_tensor = img_tensor.swapaxes(0, 2)  # can only receive two parameters
                            img_tensor = img_tensor.swapaxes(2, 1)

                            # tensor 转化为图片
                            def tensor_to_PIL(tensor):
                                image = tensor.cpu().clone()
                                image = image.squeeze(0)
                                unloader = transforms.ToPILImage()
                                image = unloader(image)
                                return image

                            img = tensor_to_PIL(img_tensor)
                            # img.show()
                            if num % 100 == 1:
                                img.save(rf'{save_dir}/{num}.jpg')
                            num += 1

                            Result_clip = detectClip(img_tensor)
                            # 在图片上绘制标签
                            label = f'{names[int(cls)]}:{conf * 100:.2f}%' \
                                    + f' {Result_clip}'  # 细分类结果：类别+概率
                            print(f'\nyolov5:{names[int(cls)]}:{conf * 100:.2f}% clip:{Result_clip}')

                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)  # TODO new1

                        else:
                            # label = f'{names[int(cls)]} {conf:.2f}' # TODO original
                            pass  # TODO new1

                        # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)# TODO original

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
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

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    def Input():
        while True:
            Input_path = str(input(rf'under detected file path[default: input/videos/test2.mp4]:'))
            if Input_path == '':
                Input_path = f'input/TEST_VIDEOS/Rick.mp4'
            elif Input_path == '1':
                Input_path = f'input/TEST_VIDEOS/R_M.mp4'
            elif Input_path == '2':
                Input_path = f'input/TEST_VIDEOS/FX.mp4'
            print(f'Input path:{Input_path}')
            return Input_path


    Input_path = Input()
    print(f'Input video:{Input_path}')

    parser = argparse.ArgumentParser()

    # TODO weights path: 1. result/anime_face/exp3/weights/best.pt , 2. pretrained/yolov5x6.pt
    parser.add_argument('--weights', nargs='+', type=str, default='result/anime_face/exp3/weights/best.pt', help='model.pt path(s)')

    parser.add_argument('--source', type=str, default=Input_path, help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.30, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='output/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
