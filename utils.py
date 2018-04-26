from __future__ import print_function

import os
import sys
import re
import shutil
import random
import cv2
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def process_data(dir, num_cls, num_per_cls, val_num):
    '''
    split data set into train and 
    '''
    if not os.path.exists(dir):
        print('Error: dir not exists')
        return

    data_dir = dir + os.path.sep + 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    # os.chdir(data_dir) # 切换目录
    train_dir = data_dir + os.path.sep + 'train'
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    val_dir = data_dir + os.path.sep + 'val'
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    # 在train目录中创建每个类别的目录
    train_sub_dirs, val_sub_dirs = create_sub_dirs_ID(
        num_cls, train_dir, val_dir)

    # 删除目录, 只保留文件
    i = 0
    f_names = os.listdir(dir)
    for item in f_names:
        if os.path.isdir(os.path.join(dir, item)):
            f_names.remove(item)

    print(len(f_names))
    # f_names = ['s165.bmp', 's17.bmp', 's18.bmp']
    f_names.sort(key=lambda x: int(re.match(r's(\d+)\.bmp', x).group(1)))
    for f in f_names:
        f_name = os.path.join(dir, f)
        if os.path.isfile(f_name):
            # print(f_name)
            if os.path.splitext(f_name)[1] == '.bmp':
                img = cv2.imread(f_name, cv2.IMREAD_UNCHANGED)
                file_cvt = os.path.splitext(f_name)[0] + '.png'
                cv2.imwrite(file_cvt, img)  # 写入png文件
                for j in range(num_cls):
                    if int(i / num_per_cls) == j:
                        if (i % num_per_cls) < (num_per_cls - val_num):
                            f_move = train_sub_dirs[j] + os.path.sep + \
                                os.path.split(os.path.realpath(file_cvt))[1]
                            if not os.path.exists(f_move):
                                shutil.move(file_cvt, train_sub_dirs[j])
                                break
                        else:
                            f_move = val_sub_dirs[j] + os.path.sep + \
                                os.path.split(os.path.realpath(file_cvt))[1]
                            if not os.path.exists(f_move):
                                shutil.move(file_cvt, val_sub_dirs[j])
                                break
                i += 1


def create_sub_dirs_ID(num_cls, train_dir, val_dir):
    train_sub_dirs = []
    val_sub_dirs = []
    for i in range(num_cls):
        train_sub_dir = train_dir + os.path.sep + str(i)
        if not os.path.exists(train_sub_dir):
            os.makedirs(train_sub_dir)
        train_sub_dirs.append(train_sub_dir)
        val_sub_dir = val_dir + os.path.sep + str(i)
        if not os.path.exists(val_sub_dir):
            os.makedirs(val_sub_dir)
        val_sub_dirs.append(val_sub_dir)
    return train_sub_dirs, val_sub_dirs


def create_sub_dirs_Name(cls_names, train_dir, val_dir):
    train_sub_dirs = []
    val_sub_dirs = []
    for i in range(len(cls_names)):
        train_sub_dir = train_dir + os.path.sep + cls_names[i]
        if not os.path.exists(train_sub_dir):
            os.makedirs(train_sub_dir)
        train_sub_dirs.append(train_sub_dir)
        val_sub_dir = val_dir + os.path.sep + cls_names[i]
        if not os.path.exists(val_sub_dir):
            os.makedirs(val_sub_dir)
        val_sub_dirs.append(val_sub_dir)
    return train_sub_dirs, val_sub_dirs


def cvt2pngs(dir):
    if not os.path.exists(dir):
        print('Error: dir not exists.')
        return
    files = os.listdir(dir)
    for f in files:
        name = os.path.join(dir, f)
        if os.path.isdir(name):
            cvt2pngs(name)
        else:
            if os.path.split(f)[1] != '.png':
                img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
                if img.empty():
                    continue
                cv2.imshow('img', img)
                cv2.waitKey(0)
                f_cvt = os.path.splitext(f)[0] + '.png'
                cv2.imwrite(f_cvt, img)
            else:
                continue


def cvt2png(dir):
    '''
    批量转换成png
    '''
    if not os.path.exists(dir):
        print('Error: dir not exists.')
        return

    g = os.walk(dir)
    for path, dir_names, file_list in tqdm(g):
        # print(dir_names)
        for file_name in tqdm(file_list):
            f = os.path.join(path, file_name)
            # print(f)
            if os.path.split(f)[1] != '.png':
                img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
                # cv2.imshow('img', img)
                # cv2.waitKey(0)
                f_cvt = os.path.splitext(f)[0] + '.png'
                cv2.imwrite(f_cvt, img)


def filter_dirs(dir, limit=4):
    '''
    过滤数据集，统计人脸不少于limit个的目录
    '''
    dirs = []
    items = os.listdir(dir)
    for item in tqdm(items):
        sub_dir_path = os.path.join(dir, item)
        if os.path.isdir(sub_dir_path):
            if len(os.listdir(sub_dir_path)) >= limit:
                # print('-- no less than %d: ' %limit, sub_dir_path)
                dirs.append(sub_dir_path)
    print('total %d dir meet requirements.' % len(dirs))
    return dirs


def filter_move(dir, limit):
    '''
    filter for 
    '''
    if not os.path.exists(dir):
        print('Error: invalid dir.')
        return

    sub_dirs = filter_dirs(dir, limit)
    cls_names = [os.path.split(sub_dir)[1] for sub_dir in sub_dirs]
    # print(cls_names)
    # print(dirs)

    # 上一级父目录
    parent_dir = os.path.abspath(os.path.join(dir, ".."))
    print(parent_dir)
    train_dir = os.path.join(parent_dir + os.path.sep + 'train')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    val_dir = os.path.join(parent_dir + os.path.sep + 'val')
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    # 根据符合条件的目录数决定分类数量，创建子目录
    train_sub_dirs, val_sub_dirs = create_sub_dirs_Name(
        cls_names, train_dir, val_dir)
    # print(train_sub_dirs)

    # 生成训练，验证数据集
    for i in range(len(cls_names)):
        train_sub_items = [os.path.join(sub_dirs[i], x)
                           for x in os.listdir(sub_dirs[i])[:14]]
        val_sub_items = [os.path.join(sub_dirs[i], x)
                         for x in os.listdir(sub_dirs[i])[14:20]]

        # 批量拷贝
        for train_item in train_sub_items:
            shutil.copy(train_item, train_sub_dirs[i])
        for val_item in val_sub_items:
            shutil.copy(val_item, val_sub_dirs[i])


def format_for_triplet(dir, limit=20, person_per_batch=5, img_per_person=10):
    '''
    prepare dataset in the triplet format
    '''
    np.random.seed(100)  # 设置固定的随机数种子,便于验证
    # 随机选择一个反例id

    def get_negative_id(sub_dirs, i):
        candidates = [j for j in range(len(sub_dirs)) if j != i]
        negative_id = np.random.choice(candidates, size=1)[0]
        return negative_id

    if not os.path.exists(dir):
        print('Error: invalid dir.')
        return

    sub_dirs = filter_dirs(dir, limit)  # 筛选不少于limit张的dir
    remain = int(len(sub_dirs) % person_per_batch)
    sub_dirs = sub_dirs[:-remain]  # batch数据对齐
    print('total: ', len(sub_dirs))

    parent_dir = os.path.abspath(os.path.join(dir, ".."))
    triplet_dir = os.path.join(parent_dir + os.path.sep + 'triplet_data')
    if not os.path.exists(triplet_dir):
        os.makedirs(triplet_dir)

    # 遍历符合要求sub_dir
    for i, sub_dir in enumerate(sub_dirs):
        # 处理每个batch
        if i % person_per_batch == 0:
            batch_id = int(i / person_per_batch)
            batch_sub_ids = [i + person_per_batch -
                             k for k in range(person_per_batch, 0, -1)]
            print('batch_id: ', batch_id, ', batch_sub_ids: ', batch_sub_ids)
            for sub_id in batch_sub_ids:
                # anchors
                anchors = [os.path.join(sub_dirs[sub_id], x)
                           for x in os.listdir(sub_dirs[sub_id])[:img_per_person]]
                for item in anchors:
                    shutil.copy(item, triplet_dir)

                # positive
                positives = [os.path.join(sub_dirs[sub_id], x)
                             for x in os.listdir(sub_dirs[sub_id])[img_per_person:limit]]
                for item in positives:
                    shutil.copy(item, triplet_dir)

                # negative
                negative_id = get_negative_id(sub_dirs, sub_id)
                negatives = [os.path.join(sub_dirs[negative_id], x)
                             for x in os.listdir(sub_dirs[negative_id])[:img_per_person]]
                for item in negatives:
                    shutil.copy(item, triplet_dir)
                pass

# def select_triplet(dir, limit=20):
#     '''
#     从数据及随机选择一个三元组
#     '''
#     np.random.seed(100)  # 设置固定的随机数种子,便于验证
#     # 随机选择一个反例id

#     def get_negative_id(length, i):
#         candidates = [j for j in range(length) if j != i]
#         negative_id = np.random.choice(candidates, size=1)[0]
#         return negative_id

#     if not os.path.exists(dir):
#         print('Error: invalid dir.')
#         return

#     sub_dirs = filter_dirs(dir, limit)  # 筛选不少于limit张的dir

#     # 随机选择一个anchor, positive, negative实例三元组
#     anchor_cls = np.random.choice(len(sub_dirs))
#     negative_cls = get_negative_id(len(sub_dirs), anchor_cls)
#     print('anchor_cls: %d, negative_cls: %d' %(anchor_cls, negative_cls))
#     anchor_id = np.random.choice(range(len(os.listdir(sub_dirs[anchor_cls]))))
#     positive_id = get_negative_id(len(os.listdir(sub_dirs[anchor_cls])), anchor_id)
#     # print('anchor_id: %d, positive id: %d' %(anchor_id, positive_id))
#     negative_id = np.random.choice(range(len(os.listdir(sub_dirs[negative_cls]))))
#     print('negative_id: ', negative_id)

#     anchor_path = os.path.join(sub_dirs[anchor_cls], os.listdir(sub_dirs[anchor_cls])[anchor_id])
#     print('anchor_path: ', anchor_path)
#     positive_path = os.path.join(sub_dirs[anchor_cls], os.listdir(sub_dirs[anchor_cls])[positive_id])
#     print('positive_path: ', positive_path)
#     negative_path = os.path.join(sub_dirs[negative_cls], os.listdir(sub_dirs[negative_cls])[negative_id])
#     print('negative_path: ', negative_path)


def move_triplet(dir, limit=20):
    '''
    创建三元组数据集
    '''
    np.random.seed(100)  # 设置固定的随机数种子,便于验证
    # 随机选择一个反例id

    def get_negative_id(length, i):
        candidates = [j for j in range(length) if j != i]
        negative_id = np.random.choice(candidates, size=1)[0]
        return negative_id

    if not os.path.exists(dir):
        print('Error: invalid dir.')
        return

    parent_dir = os.path.abspath(os.path.join(dir, ".."))
    triplet_dir = os.path.join(parent_dir + os.path.sep + 'triplet_data')
    if not os.path.exists(triplet_dir):
        os.makedirs(triplet_dir)

    sub_dirs = filter_dirs(dir, limit)  # 筛选不少于limit张的dir
    for sub_dir in tqdm(sub_dirs):
        for item in os.listdir(sub_dir)[:limit]:
            shutil.copy(os.path.join(sub_dir, item), triplet_dir)


# 随机选择一个反例id
def get_negative_id(length, i):
    candidates = [j for j in range(length) if j != i]
    negative_id = np.random.choice(candidates, size=1)[0]
    return negative_id


def select_triplet(dir, num_classes, limit=20, is_car=False):
    '''
    从triplet_dir中随机选择一个三元组
    '''
    # np.random.seed(100)  # 设置固定的随机数种子,便于验证

    # 获取anchor, positive, negative 图片ID
    anchor_cls = np.random.choice(num_classes)
    anchor_id = anchor_cls*limit + np.random.choice(limit)
    positive_id = anchor_cls*limit + \
        get_negative_id(limit, anchor_id - anchor_cls*limit)
    negative_cls = get_negative_id(num_classes, anchor_cls)  # 随机选择一个反例类型
    negative_id = negative_cls*limit + np.random.choice(limit)  # 随机选择一个反例ID

    # 获取anchor, positive, negative 图片地址
    file_names = os.listdir(dir)
    if is_car:  # 汽车图片被重命名过,因此需要按照(数值大小)从小到大排序
        file_names.sort(key=lambda x: int(re.match(r'(\d+)\.jpg', x).group(1)))

    anchor_path = os.path.join(dir, file_names[anchor_id])
    positive_path = os.path.join(dir, file_names[positive_id])
    negative_path = os.path.join(dir, file_names[negative_id])

    # 返回6元组
    return anchor_path, positive_path, negative_path, anchor_cls, anchor_cls, negative_cls


def select_car_triplet(dir, labels, label2img):
    '''
    从triplet_dir中随机选择一个三元组: 图像并没有重命名
    '''
    anchor_cls = np.random.choice(len(labels))
    anchor_id = np.random.choice(len(label2img[labels[anchor_cls]]))
    positive_id = get_negative_id(
        len(label2img[labels[anchor_cls]]), anchor_id)
    negative_cls = get_negative_id(len(labels), anchor_cls)
    negative_id = np.random.choice(len(label2img[labels[negative_cls]]))

    # anchor, positive, negative文件路径
    anchor_path = os.path.join(dir, label2img[labels[anchor_cls]][anchor_id])
    positive_path = os.path.join(
        dir, label2img[labels[anchor_cls]][positive_id])
    negative_path = os.path.join(
        dir, label2img[labels[negative_cls]][negative_id])

    return anchor_path, positive_path, negative_path, \
        labels[anchor_cls], labels[anchor_cls], labels[negative_cls]


def get_all_triplets(dir, limit=20):
    '''
    计算所有的triplet
    '''
    if not os.path.exists(dir):
        print('Error: invalid dir.')
        return

    sub_dirs = filter_dirs(dir, limit)  # 筛选不少于limit张的dir

    triplets = []
    for anchor_dir in tqdm(sub_dirs):
        anchor_paths = [os.path.join(anchor_dir, x)
                        for x in os.listdir(anchor_dir)]
        for anchor_path in anchor_paths:
            for positive_path in [path for path in anchor_paths if path != anchor_path]:
                # print('anchor_path: %s, positive_path: %s' %(anchor_path, positive_path))
                for negative_dir in [the_dir for the_dir in sub_dirs if the_dir != anchor_dir]:
                    negative_paths = [os.path.join(
                        negative_dir, x) for x in os.listdir(negative_dir)]
                    for negative_path in negative_paths:
                        anchor_label = sub_dirs.index(anchor_dir)
                        negative_label = sub_dirs.index(negative_dir)
                        triplet = anchor_path, positive_path, negative_path, \
                            anchor_label, anchor_label, negative_label
                        print(triplet)
                        triplets.append(triplet)
    # print(triplets)


# 为测试从dataset输入数据集的方法, 生成所有数据在一个目录的文件夹
def prepare_for_dataset(dir, limit=20):
    if not os.path.exists(dir):
        print('Error: invalid dir.')
        return

    parent_dir = os.path.abspath(os.path.join(dir, ".."))
    my_dataset = os.path.join(parent_dir + os.path.sep + 'my_dataset')
    if not os.path.exists(my_dataset):
        os.makedirs(my_dataset)

    sub_dirs = filter_dirs(dir, limit)  # 筛选不少于limit张的dir
    for sub_dir in tqdm(sub_dirs):
        for f in os.listdir(sub_dir)[:limit]:
            shutil.copy(os.path.join(sub_dir, f), my_dataset)


# -------------创建验证数据集
def create_validate_set(dir, limit=20, num=100):
    '''
    @param dirs: valid dirs with enough examplars
    基本要求: 数据与label对应起来
    '''
    if not os.path.exists(dir):
        print('Error: invalid dir.')
        return

    dirs = filter_dirs(dir, limit)  # 筛选不少于limit张的dir
    dirs.sort()
    if len(dirs) == 0:
        print('Error: invalid dirs.')
        return
    print('mapping:\n')
    for label, name in enumerate(dirs):
        print('[%d, %s]' % (label, name))

    # 创建validation set目录
    parant_dir = os.path.abspath(os.path.join(dir, '..'))
    validset_path = os.path.join(parant_dir + os.path.sep + 'validate_set')
    print(validset_path)
    if os.path.exists(validset_path):
        shutil.rmtree(validset_path)  # 如果已经存在, 清空目录
        os.makedirs(validset_path)
    else:
        os.makedirs(validset_path)

    data_label = {}
    for i in tqdm(range(num)):
        dir_id = np.random.choice(len(dirs))

        # 生成validation数据集
        items = os.listdir(dirs[dir_id])
        item_id = np.random.choice(len(items))
        item_path = os.path.join(dirs[dir_id], items[item_id])
        dst_path = os.path.join(validset_path, os.path.split(item_path)[1])
        while os.path.exists(dst_path):  # 如果已经存在于validset就重新选择一个
            item_id = np.random.choice(len(items))
            item_path = os.path.join(dirs[dir_id], items[item_id])
            dst_path = os.path.join(validset_path, os.path.split(item_path)[1])
        data_label[items[item_id]] = dir_id
        shutil.copy(item_path, validset_path)

    # 保存label文件
    labels = [v for k, v in data_label.items()]
    labels.sort()
    print(labels)
    label_path = os.path.join(validset_path, 'labels.txt')
    if os.path.exists(label_path):
        os.remove(label_path)
    with open(label_path, 'w') as f:
        for label in labels:
            f.write(str(label) + '\n')


def get_validate_set(dir, num_per_cls=10, limit=20):
    '''
    创建验证数据集
    '''
    if not os.path.exists(dir):
        print('Error: invalid dir.')
        return

    dirs = filter_dirs(dir, limit)  # 筛选不少于limit张的dir
    dirs.sort()
    if len(dirs) == 0:
        print('Error: invalid dirs.')
        return
    # print('mapping:\n')
    # for label, name in enumerate(dirs):
    #     print('[%d, %s]' %(label, name))

    # 创建validation set目录
    parant_dir = os.path.abspath(os.path.join(dir, '..'))
    validset_path = os.path.join(parant_dir + os.path.sep + 'validate_set')
    print(validset_path)
    if os.path.exists(validset_path):
        shutil.rmtree(validset_path)  # 如果已经存在, 清空目录
        os.makedirs(validset_path)
    else:
        os.makedirs(validset_path)

    # 创建validate_set数据集
    for i_, dir_ in tqdm(enumerate(dirs)):
        for i in range(num_per_cls):  # 每个类别数量相同
            items = os.listdir(dir_)
            item_id = np.random.choice(len(items))  # 随机选择一个该类别的样本
            item_path = os.path.join(dir_, items[item_id])
            dst_path = os.path.join(validset_path, os.path.split(item_path)[1])
            while os.path.exists(dst_path):  # 如果已经存在于validset就重新选择一个
                item_id = np.random.choice(len(items))
                item_path = os.path.join(dir_, items[item_id])
                dst_path = os.path.join(
                    validset_path, os.path.split(item_path)[1])
            shutil.copy(item_path, validset_path)


# 读取.mat标签文件
import scipy.io as scio
from collections import defaultdict


def read_class_names(path):
    '''
    @param path: meta mat path
    '''
    if not os.path.exists(path):
        print('Error: invalid path.')
        return

    data = scio.loadmat(path)
    # print(data.keys(), '\n')
    # print('__header__: ', data['__header__'])
    # print('__version__: ', data['__version__'])
    # print('__global__: ', data['__globals__'])

    class_names = [x[0] for x in data['class_names'][0]]
    # print(class_names)
    return class_names  # 依据car的label(int)索引


def read_imgname_label(path):
    '''
    @param path: mat文件path
    '''
    if not os.path.exists(path):
        print('Error: invalid path.')
        return

    data = scio.loadmat(path)
    annots = data['annotations']
    annots = annots[0]

    img_names = [str(x[-1][0]) for x in annots]
    img_labels = [x[-2][0][0] for x in annots]
    # print(img_labels)

    id_imgs = {}
    img_name2labels = {}
    label_img_names = defaultdict(list)
    for id, (name, label) in enumerate(zip(img_names, img_labels)):
        # print('%d, %s, %d' %(id, name, label))
        id_imgs[id] = (name, label)  # 数据集中第id张图的信息

        img_name2labels[name] = label  # 记录每个文件名对应的label
        label_img_names[label].append(name)  # 记录每个label对应所有文件名
        # print('id: %d, img_name: %s, label: %d' %(id, id_imgs[id][0], id_imgs[id][1]))
    return id_imgs, img_labels, img_name2labels, label_img_names


import pickle


def process_cars(meta_path, annot_path, orig_train_dir, limit=40):
    '''
    根据元信息和annotations读取labels
    '''
    class_names = read_class_names(meta_path)
    print('total %d kind of cars.' % len(class_names))

    id_imgs, labels, img_names2label, label2img_names = read_imgname_label(
        annot_path)
    print(len(img_names2label))  # 所有训练数据集, 名字->label映射

    # for (k, v) in id_imgs.items():
    #     print('file_%d, img_name: %s, label: %d, class_name: %s' %
    #           (k, v[0], v[1], class_names[v[1] - 1])) # 从class label转换成label索引,需要-1

    # 统计训练数据及中196种车型各自的图片数量
    freqs = defaultdict(int)
    for label in labels:
        freqs[label] += 1
    sorted_freqs = sorted(freqs.items(), key=lambda label: label[1])
    # print(sorted_freqs)  # 按频数从小到大排序

    # 创建训练目录
    train_dir = os.path.join(os.getcwd(), 'car_train_data')
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)  # 如果已经存在, 清空目录,重新创建目录
        os.makedirs(train_dir)
    else:
        os.makedirs(train_dir)

    labels = sorted(set(labels))
    # print(labels) # 1~196
    # 获取label到类别名称的映射
    label2_class_names = defaultdict(str)
    for label in labels:
        # label转换成label的索引, 需要-1
        label2_class_names[label] = class_names[label - 1]
        # print(label2_class_names[label])

    # 将数量>limit的车copy到一个训练目录
    qualified_labels = []
    # new2old_lables = defaultdict(int)
    new_lable2classname = defaultdict(str)
    i = 0
    for label in tqdm(labels):
        if freqs[label] >= limit:  # 满足数量要求的图片
            qualified_labels.append(label)
            for img_file in label2img_names[label]:
                shutil.copy(os.path.join(orig_train_dir, img_file), train_dir)

            # -----------------------------------------
            # # 每个类别, 只拷贝limit个, 拷贝并重命名
            # item_ids = np.random.choice(len(label2img_names[label]), size=limit)
            # for j, item_id in enumerate(item_ids):
            #     start_id = i*limit
            #     dst_path = os.path.join(train_dir, str(start_id + j) + '.jpg')
            #     src_path = os.path.join(
            #         orig_train_dir, label2img_names[label][item_id])
            #     shutil.copyfile(src_path, dst_path)

            #     # 判断通道数, 将1通道数据转换成RGB标准3通道
            #     img = cv2.imread(dst_path, cv2.IMREAD_UNCHANGED)
            #     num_channels = len(cv2.split(img))
            #     if num_channels == 1:
            #         img_cvt = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            #         cv2.imwrite(dst_path, img_cvt)
            #         print('converted channel number: %d,' % len(cv2.split(img_cvt)), end='')
            #         print(' converted from 1 channel to 3 channels.')
            # # new2old_lables[i] = label
            # new_lable2classname[i] = label2_class_names[label]  # 车类别的实际名称
            # # print('%d mapping to %d: %s' % (i, label, new_lable2classname[i]))
            # -----------------------------------------
            i += 1

    # 预处理每一个类别的数据: 将只有1通道的图像转换成RGB标准3通道
    files = os.listdir(train_dir)
    for f in files:
        f_path = os.path.join(train_dir, f)
        img = cv2.imread(f_path, cv2.IMREAD_UNCHANGED)
        if len(cv2.split(img)) == 1:
            print('%s is 1 channel image, ' % f, end='')
            img_cvt = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            cv2.imwrite(f_path, img_cvt)
            print('%s is converted to %d channels.' %
                  (f, len(cv2.split(img_cvt))))
    print('total %d kind of cars meet requirements.' % i)
    # print('qualified labels:\n', qualified_labels)

    # 验证正确性
    # for label in qualified_labels:
    #     print('\nqualifieed class_name: {}'.format(label2_class_names[label]))
    #     print([label2img_names[label]])

    # 计算每个qualified label到改label对应所有文件名的映射
    qual_label2img_names = defaultdict(list)
    for label in qualified_labels:
        qual_label2img_names[label] = label2img_names[label]

    # 将类别标签, 类别名称等信息序列化到硬盘
    label_dict_file = os.path.join(os.getcwd(), 'img_names2labels.pkl')
    qualified_label_file = os.path.join(os.getcwd(), 'quali_labels.pkl')
    qual_label2img_names_file = os.path.join(
        os.getcwd(), 'label2img_names.pkl')
    class_names_file = os.path.join(os.getcwd(), 'class_names.pkl')
    new_lable2classname_file = os.path.join(
        os.getcwd(), 'new_lable2classname.pkl')

    pickle.dump(qualified_labels, open(
        qualified_label_file, 'wb'))  # 记录有效的label
    pickle.dump(img_names2label, open(
        label_dict_file, 'wb'))  # 记录每个文件名对应的label
    pickle.dump(qual_label2img_names, open(
        qual_label2img_names_file, 'wb'))  # 记录每个有效label对应的所有文件名
    pickle.dump(class_names, open(class_names_file, 'wb'))
    pickle.dump(new_lable2classname, open(new_lable2classname_file, 'wb'))
    return qualified_labels, img_names2label


# def process_car(meta_path, annot_path, orig_train_dir, limit=40):
#     class_names = read_class_names(meta_path)
#     print('total %d kind of cars.' % len(class_names))

#     id_imgs, labels, img_names2label, label2img_names = read_imgname_label(
#         annot_path)

#     #

def test_car_triplet():
    '''
    测试car triplet选择是否正确
    '''
    quali_labels = pickle.load(open('quali_labels.pkl', 'rb'))
    img_name2labels = pickle.load(open('img_names2labels.pkl', 'rb'))
    label2img_names = pickle.load(open('label2img_names.pkl', 'rb'))
    class_names = pickle.load(open('class_names.pkl', 'rb'))

    fig = plt.figure()
    for i in range(3):
        a, b, c, d, e, f = select_car_triplet('car_train_data',
                                              quali_labels,
                                              label2img_names)
        print('{}, {}, {}'.format(
            class_names[d-1], class_names[e-1], class_names[f-1]))

        anchor = Image.open(a)
        positive = Image.open(b)
        negative = Image.open(c)

        anchor = anchor.resize((400, 400))
        positive = positive.resize((400, 400))
        negative = negative.resize((400, 400))

        ax_0 = plt.subplot(131)
        ax_1 = plt.subplot(132)
        ax_2 = plt.subplot(133)

        ax_0.imshow(anchor)
        ax_1.imshow(positive)
        ax_2.imshow(negative)
        plt.show()


def resize_imgs(dir, width, height):
    '''
    将目录下所有图片缩放到指定大小
    '''
    if not os.path.exists(dir):
        print('Error: invalid dir.')
        return

    f_names = os.listdir(dir)
    # print(f_names)

    for f in tqdm(f_names):
        f_name = os.path.join(dir, f)
        if os.path.isfile(f_name):
            if os.path.splitext(f_name)[1] == '.jpg':
                img = cv2.imread(f_name, cv2.IMREAD_UNCHANGED)
                if img.shape[0] != height and img.shape[1] != width:
                    img = cv2.resize(img, (height, width),
                                     interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(f_name, img)  # 写入所犯之后的图片


def judge_channel(dir):
    '''
    判断一个目下的图像是灰度图还是彩色图
    '''
    if not os.path.exists(dir):
        print('Error: invalid dir.')
        return

    for f in os.listdir(dir):
        f_path = os.path.join(dir, f)
        if os.path.isfile(f_path):
            img = cv2.imread(f_path, cv2.IMREAD_UNCHANGED)
            channels = cv2.split(img)
            if len(channels) == 1:
                print('%s is 1 channel image,' % (f_path), end='')
                img_cvt = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                cns = cv2.split(img_cvt)
                cv2.imwrite(f_path, img_cvt)
                print(' converted {} channels.'.format(len(cns)))
                # cv2.imshow('3 channel', img)
                # cv2.waitKey()
            # elif len(channels) == 3:
            #     print('%s is 3 channel image.' %(f_path))
            elif len(channels) == 4:
                print('%s is 4 channel image.' % (f_path))


# 从人脸训练数据集中随机生成一个测试数据集
def get_test_from_train(dir, num_classes=62, num_per_cls=20):
    if not os.path.exists(dir):
        print('Error: invalid dir.')
        return
    parent_dir = os.path.realpath(os.path.join(dir, '..'))
    test_set = os.path.join(parent_dir + os.path.sep + 'test_set')
    if os.path.exists(test_set):
        shutil.rmtree(test_set)
        os.makedirs(test_set)
    else:
        os.makedirs(test_set)

    files = os.listdir(dir)
    for i in tqdm(range(num_classes)):
        ids = np.random.choice(num_per_cls,
                               int(num_per_cls * 0.5),
                               replace=False)  # 不放回抽取
        start_id = i * num_per_cls
        for id in ids:
            id += start_id
            src_path = os.path.join(dir, files[id])
            shutil.copy(src_path, test_set)

# off-line all hard triplets mining
def get_all_hard_triplets(dir, model_path,
                          num_classes=62, num_per_cls=20):
    '''
    在train_set中offline hard sample mining的方式选择所有triplets
    '''
    # 加载特征提取CNN模型
    model = torch.load(model_path)

    # 将全连接层分类器转换成为恒等映射, 用来提取深度特征
    del model._classifier
    model._classifier = lambda x: x
    model.cuda()  # 模型放进GPU
    model.eval()  # 求值模式

    # 训练数据集所有文件名
    files = os.listdir(dir)

    # 图像处理方式
    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=np.array([0.485, 0.456, 0.406]),
            std=np.array([0.229, 0.224, 0.225])),
    ])

    triplets = []  # 存放所有被选择的triplet

    # 遍历每一个类别
    for anchor_cls in tqdm(range(num_classes)):
        positive_cls = anchor_cls

        # 随机选择一个正样本作为anchor:不应该随机选择,应该遍历所有id
        for id in range(num_per_cls):
            anchor_id = anchor_cls * num_per_cls + id # np.random.choice(num_per_cls)
            anchor_path = os.path.join(dir, files[anchor_id])
            anchor_img = Variable(transform(Image.open(anchor_path)).unsqueeze(0),
                                  volatile=True).cuda()  # 数据放入GPU
            anchor_feats = model.forward(anchor_img)
            anchor_feats = torch.div(
                anchor_feats, torch.norm(anchor_feats, 2))  # 特征向量L2规范化
            # print('anchor_feats:\n', anchor_feats)

            # 遍历每一个负样本选择最hard的一个
            negative_cls = -1
            negative_id = -1
            negative_path = ''
            dist = float('inf')  # 无穷大
            for nega_cls in range(num_classes):# 遍历每一个负样本类别
                if nega_cls == anchor_cls:
                    continue
                nega_start = nega_cls * num_per_cls
                # 遍历所有负样本的类别的所有负样本文件
                for nega_id in range(nega_start, nega_start + num_per_cls):
                    nega_path = os.path.join(dir, files[nega_id])
                    nega_img = Variable(transform(Image.open(nega_path)).unsqueeze(0),
                                        volatile=True).cuda()  # 数据放入GPU
                    nega_feats = model.forward(nega_img)

                    # 特征L2张量规范化
                    nega_feats = torch.div(nega_feats, torch.norm(nega_feats, 2))

                    # 计算L2规范化后的特征向量L2距离
                    nega_anchor_dist = F.pairwise_distance(
                        nega_feats, anchor_feats, p=2).cpu().data.numpy()[0]
                    if nega_anchor_dist < dist:
                        dist = nega_anchor_dist
                        negative_cls = nega_cls
                        negative_id = nega_id

            # 计算该ID的每一对anchor——positive正样本对
            if negative_id != -1:
                negative_path = os.path.join(dir, files[negative_id])
                positive_ids = [positive_cls * num_per_cls +
                                id for id in range(num_per_cls)]
                positive_ids = [id for id in positive_ids if id != anchor_id]
                for positive_id in positive_ids:
                    positive_path = os.path.join(dir, files[positive_id])
                    triplets.append((anchor_path, positive_path, negative_path,
                                     anchor_cls, positive_cls, negative_cls))
            else:
                print('invalid negative sample.')
    print(triplets)
    print('total %d triplets.' % len(triplets))

    # 应该把这个triplets存到硬盘
    pickle.dump(triplets, open('triplets.pkl', 'wb'))
    return triplets


if __name__ == '__main__':
    # cvt2png('./YaleFace')
    # process_data('./YaleFace', 15, 11, 4)
    # filter_dirs('lfw', 20)
    # prepare_for_dataset('lfw', 20)
    # filter_move('lfw', 20)
    # format_for_triplet('lfw')
    # select_triplet('lfw')
    # move_triplet('lfw')
    # select_triplet('triplet_data', 62)
    # get_all_triplets('lfw')
    # get_validate_set('lfw')
    # read_imgname_label('car_devkit/cars_train_annos.mat')
    # read_class_names('car_devkit/cars_meta.mat')
    # process_cars('car_devkit/cars_meta.mat',
    #              'car_devkit/cars_train_annos.mat',
    #              'cars_train')
    # resize_imgs('car_train_data', 250, 250)
    # test_car_triplet()
    # select_car_triplet('car_train_data', 196, limit=20)
    # judge_channel('car_train_data')
    # get_test_from_train('train_set')
    get_all_hard_triplets('train_set', 'checkpoints/epoch_35.pth')
    print('\n--Test done.')


# pytorch数据集的准备工作
# http://www.bubuko.com/infodetail-2304938.html
