import os
import re
import pickle
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
from utils import select_triplet, select_car_triplet, get_all_hard_triplets
from tqdm import tqdm


class Faces_YALE(data.Dataset):
    def __init__(self, root, num_per_cls=11,
                 transforms=None, train=True, test=False):
        '''
        获取图像路径，并根据训练，验证，测试划分数据
        '''
        self.test = test  # 是否是测试模式

        # 获取文件路径
        print('root: ', root)
        imgs = [os.path.join(root, img) for img in os.listdir(root)]

        # 文件名排序
        imgs = sorted(imgs, key=lambda x: int(
            re.match(r'.*s(\d+)\.png', x).group(1)))
        print(imgs)

        img_num = len(imgs)

        # 随机打乱数据
        # np.random.seed(100) # 设置随机种子，确定
        # imgs = np.random.permutation(imgs) # Yale数据集暂时不能随机打乱

        # 划分训练数据集和验证数据集7: 3
        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.7 * img_num)]  # 前70%
        else:
            self.imgs = imgs[int(0.7 * img_num):]  # 后30%

        # 数据中心化
        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

            # 测试数据集或者验证数据集无需数据增强
            if self.test or train:  # or not train
                self.transforms = T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            else:
                self.transforms = T.Compose([
                    T.Resize(256),
                    T.RandomSizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        '''
        一次返回一张图片的数据和label
        '''
        img_path = self.imgs[index]
        face_id = int(re.match(r'.*s(\d+)\.png', img_path).group(1))
        # print('face id: ', face_id)
        label = int(face_id / 11)
        data = Image.open(img_path)
        data = self.transforms(data)  # 对数据进行缩放、裁剪、增强，归一化的变换
        return data, label

    def __len__(self):
        return len(self.imgs)


class FACE_LFW(data.Dataset):
    '''
    处理LYW数据集, 通过自定义dataset读取数据
    '''

    def __init__(self,
                 root,
                 transforms=None,
                 NUM_PER_CLS=20):
        '''
        初始化数据集
        '''
        self.num_per_cls = NUM_PER_CLS

        # 获取文件路径
        print('root: ', root)
        imgs = [os.path.join(root, img) for img in os.listdir(root)]
        self.imgs = imgs

        # 数据中心化
        if transforms is None:
            self.transforms = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])])

        else:
            self.transforms = transforms

    def __getitem__(self, index):
        '''
        通过索引获取数据和标签
        '''
        img_path = self.imgs[index]
        label = int(index / self.num_per_cls)
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)


class Hard_Triplet(data.Dataset):
    '''
    基于FaceNet Hard Triplet
    '''
    def __init__(self, root, model_path, transform=None):
        '''
        hard triplets初始化, 数据预处理方式
        '''
        self.triplets = get_all_hard_triplets(root, model_path, 62, 20)

        # 数据预处理
        if transform == None:
            self.transform = T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])])
        else:
            self.transform = transform

    def __getitem__(self, index):
        '''
        一次获取一个hard triplet
        '''
        triplet = self.triplets[index]
        data = [self.transform(Image.open(img)) for img in triplet[:3]]
        label = triplet[3:]
        return data, label

    def __len__(self):
        '''
        返回所有hard triplet数量
        '''
        return len(self.triplets)


class Triplet(data.Dataset):
    '''
    每次随机产生一个triplet数据
    '''

    def __init__(self,
                 root,
                 num_cls=62,
                 num_tripets=1000,
                 limit=20,
                 transforms=None,
                 train=True,
                 test=False):
        '''
        组织数据: 随机产生num_triplets个triplets
        '''
        self.test = test
        self.triplets = [select_triplet(root, num_cls, limit, False)
                         for i in range(num_tripets)]  # order: anchor, positive, negative
        # print(self.triplets)

        # 数据预处理
        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

            # 测试数据集或者验证数据集无需数据增强
            if self.test or train:  # or not train
                self.transforms = T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            else:  # 数据增强
                self.transforms = T.Compose([
                    T.Resize(256),
                    T.RandomSizedCrop(224),
                    T.RandomHorizontalFlip(),
                    normalize,
                    T.ToTensor()
                ])
        else:
            self.transforms = transforms

    def __getitem__(self, index):
        '''
        每次返回一个triplet
        '''
        triplet = self.triplets[index]
        # print(triplet)
        data = [self.transforms(Image.open(img_path))
                for img_path in triplet[:3]]
        label = triplet[3:]
        # print(label)
        return data, label

    def __len__(self):
        '''
        返回triplet数量
        '''
        return len(self.triplets)

# --------------------Triplet for cars


class Car_triplet(data.Dataset):
    '''
    汽车的Triplet数据集
    '''

    def __init__(self,
                 root,
                 label_path,
                 label2img_path,
                 num_triplets=5000):
        '''
        初始化数据集
        '''
        print('root: ', root)
        self.labels = pickle.load(open(label_path, 'rb'))
        print('total %d kinds of car.' % len(self.labels))
        self.label2img = pickle.load(open(label2img_path, 'rb'))

        # 生成triplets数据集
        self.triplets = [select_car_triplet(root, self.labels, self.label2img)
                         for i in range(num_triplets)]
        # print(self.triplets)

        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

        # 数据预处理
        self.transforms = T.Compose([
            T.Compose([
                T.Resize(224),
                T.CenterCrop(224),
                T.ToTensor(),
                normalize  # normalize与ToTensor能否改变顺序
            ])
        ])

    def __getitem__(self, index):
        '''
        每次返回一个triplet
        '''
        triplet = self.triplets[index]
        data = [self.transforms(Image.open(img)) for img in triplet[:3]]
        label = triplet[3:]
        return data, label

    def __len__(self):
        '''
        返回triplet数量
        '''
        return len(self.triplets)


# Car test dataset...


# if __name__ == '__main__':
#     triplets = [select_car_triplet('car_train_data', 196) for i in tqdm(range(1000))]
#     # print(triplets)
#     for x in triplets:
#         print(x)
#     print('--Test done.')


# if __name__ == '__main__':
#     labels = pickle.load(open('quali_labels.pkl', 'rb'))
#     img2label = pickle.load(open('img_names2labels.pkl', 'rb'))
#     label2img = pickle.load(open('label2img_names.pkl', 'rb'))
#     # class_names = pickle.load(open('class_names.pkl', 'rb'))
#     triplets = [select_car_triplet('car_train_data',
#                                    labels,
#                                    img2label,
#                                    label2img) for i in range(100)]
#     print(triplets)
