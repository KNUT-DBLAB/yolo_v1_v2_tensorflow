import xml.etree.cElementTree as ET
import numpy as np
import os
from PIL import Image
import scipy.misc as misc
import scipy.io as sio
import numpy


OBJECT_NAMES = ["tvmonitor", "train", "sofa", "sheep", "cat", "chair", "bottle", "motorbike", "boat", "bird",
                   "person", "aeroplane", "dog", "pottedplant", "cow", "bus", "diningtable", "horse", "bicycle", "car"]
EPSILON = 1e-8

def read_xml(path):
    tree = ET.parse(path)
    root = tree.getroot()
    objects = root.findall("object")
    imgname = root.find("filename").text
    gt_bbox = np.zeros([objects.__len__(), 4], dtype=np.int32)
    name_bbox = []
    for i, obj in enumerate(objects):
        objectname = obj.find("name").text
        bbox = np.zeros([4], dtype=np.int32)
        xmin = int(obj.find("bndbox").find("xmin").text)
        ymin = int(obj.find("bndbox").find("ymin").text)
        xmax = int(obj.find("bndbox").find("xmax").text)
        ymax = int(obj.find("bndbox").find("ymax").text)
        bbox[0], bbox[1], bbox[2], bbox[3] = xmin, ymin, xmax, ymax
        name_bbox.append(objectname)
        gt_bbox[i, :] = bbox

    return imgname, gt_bbox, name_bbox

def cal_iou(bbox1, bbox2):
    #bbox = [x1, y1, x2, y2]
    x1, y1, x1_, y1_ = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    x2, y2, x2_, y2_ = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
    x0 = max(x1, x2)
    y0 = max(y1, y2)
    x0_ = min(x1_, x2_)
    y0_ = min(y1_, y2_)

    if x0 >= x0_ or y0 >= y0_:
        iou = 0
    else:
        inter_area = (x0_ - x0) * (y0_ - y0)
        bbox1_area = (x1_ - x1) * (y1_ - y1)
        bbox2_area = (x2_ - x2) * (y2_ - y2)
        union_area = bbox1_area + bbox2_area - inter_area

        iou = inter_area / union_area
    return iou

def ToScaleImg(img, tar_h, tar_w, raw_bboxes):
    h, w = img.shape[0], img.shape[1]
    nums_bbox = raw_bboxes.shape[0]
    tar_bboxes = np.zeros_like(raw_bboxes)
    for i in range(nums_bbox):
        bbox = raw_bboxes[i]
        x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
        x0 = tar_w / w * x0
        x1 = tar_w / w * x1
        y0 = tar_h / h * y0
        y1 = tar_h / h * y1
        tar_bboxes[i, 0], tar_bboxes[i, 1] = x0, y0
        tar_bboxes[i, 2], tar_bboxes[i, 3] = x1, y1

    # print(img)
    # print(img.shape)
    
    # print(tar_h)
    # print(tar_w)
    # scaled_img = misc.imresize(img, [tar_h, tar_w]) # https://github.com/tensorlayer/srgan/pull/179
    scaled_img = numpy.array(Image.fromarray(img).resize((tar_h, tar_w)))

    # scaled_img = sk_resize(img, (tar_h, tar_w))
    # print(tar_h, tar_w)
    
    
    return scaled_img, tar_bboxes


def read_batch(img_path, xml_path, batch_size, img_h=448, img_w=448):
    xml_lists = os.listdir(xml_path)
    nums = xml_lists.__len__()
    rand_idx = np.random.randint(0, nums, [batch_size])
    batch_bboxes = np.zeros([batch_size, 7, 7, 4])
    batch_classes = np.zeros([batch_size, 7, 7, 20])
    batch_img = np.zeros([batch_size, img_h, img_w, 3])
    cell_h = img_h / 7
    cell_w = img_w / 7
    for j in range(batch_size):
        imgname, gt_bbox, name_bbox = read_xml(xml_path + xml_lists[rand_idx[j]])
        img = np.array(Image.open(img_path + imgname))
        scaled_img, scaled_bbox = ToScaleImg(img, img_h, img_w, gt_bbox)
        batch_img[j, :, :, :] = scaled_img
        for i in range(scaled_bbox.shape[0]):
            c_x = (scaled_bbox[i, 0] + scaled_bbox[i, 2]) / 2
            c_y = (scaled_bbox[i, 1] + scaled_bbox[i, 3]) / 2
            h = scaled_bbox[i, 3] - scaled_bbox[i, 1]
            w = scaled_bbox[i, 2] - scaled_bbox[i, 0]
            col = int(c_x // cell_w)
            row = int(c_y // cell_h)
            offset_x = c_x / cell_w - col
            offset_y = c_y / cell_h - row
            offset_h = np.sqrt(h / img_h)
            offset_w = np.sqrt(w / img_w)
            batch_bboxes[j, row, col, 0], batch_bboxes[j, row, col, 1] = offset_x, offset_y
            batch_bboxes[j, row, col, 2], batch_bboxes[j, row, col, 3] = offset_h, offset_w
            index = OBJECT_NAMES.index(name_bbox[i])
            batch_classes[j, row, col, index] = 1
    batch_labels = np.zeros([batch_size, 7, 7, 25])
    batch_response = np.sum(batch_classes, axis=-1, keepdims=True)
    batch_labels[:, :, :, 0:1] = batch_response
    batch_labels[:, :, :, 1:5] = batch_bboxes
    batch_labels[:, :, :, 5:] = batch_classes
    return batch_img, batch_labels

def img2mat(imgpath, xmlpath):
    filenames = os.listdir(xmlpath)
    nums = filenames.__len__()
    imgs = np.zeros([nums, 448, 448, 3], dtype=np.uint8)
    xml = []
    class_name = []
    for idx, filename in enumerate(filenames):
        imgname, gt_bbox, name_bbox = read_xml(xmlpath + filename)
        img = np.array(Image.open(imgpath + imgname))
        scaled_img, scaled_bbox = ToScaleImg(img, 448, 448, gt_bbox)
        imgs[idx, :, :, :] = scaled_img
        xml.append(scaled_bbox)
        class_name.append(name_bbox)
        print(idx)
    sio.savemat("pascal.mat", {"imgs": imgs, "bboxes": xml, "class": class_name})

