import argparse
import os
import re
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import dataset
from cnn_finetune import make_model
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


def test_triplet():
    parser = argparse.ArgumentParser(
        description='Face recognition using triplet loss.')
    parser.add_argument('--train-set', type=str, default='train_set', metavar='T',
                        help='path of train set.')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--model-name', type=str, default='resnet18', metavar='M',
                        help='model name (default: resnet50)')
    parser.add_argument('--dropout-p', type=float, default=0.2, metavar='D',
                        help='Dropout probability (default: 0.2)')
    parser.add_argument('--check-path', type=str,
                        default='checkpoints', metavar='C', help='Checkpoint path')
    parser.add_argument('--is-resume', type=bool, default=True,
                        metavar='R', help='whether resume from latest checkpoint.')

    # ----------------------参数
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # ----------------------模型
    if args.is_resume:  # 从checkpoint恢复模型
        checkpoints = os.listdir(args.check_path)
        checkpoints.sort(key=lambda x: int(re.match(r'epoch_(\d+)\.pth', x).group(1)),
                         reverse=True)
        model = torch.load(os.path.join(
            args.check_path + os.path.sep + checkpoints[0]))
        LATEST_MODEL_ID = int(
            re.match(r'epoch_(\d+)\.pth', checkpoints[0]).group(1))
        print('[resume from model, model id: %d]' % LATEST_MODEL_ID)
    else:  # 从训练好的模型加载
        model = make_model(args.model_name,
                           pretrained=True,
                           num_classes=62,
                           dropout_p=args.dropout_p)
    # print('model:\n', model)

    if args.cuda:
        model.cuda()

    # ----------------------对图片数据处理: 转换成Tensor并中心归一化
    transform = transforms.Compose([
        # transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=np.array([0.485, 0.456, 0.406]),
            std=np.array([0.229, 0.224, 0.225])),
    ])

    # ----------------------加载训练数据集
    # train_set = dataset.Triplet(args.train_set,
    #                             num_cls=62,  # 62
    #                             num_tripets=8000,
    #                             limit=20,  # 20
    #                             transforms=transform,
    #                             train=True,
    #                             test=False)
    train_set = dataset.Hard_Triplet(args.train_set,
                                     'checkpoints/epoch_35.pth') # 先人为指定...
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=2)

    # ----------------------加载测试数据集
    test_set = dataset.FACE_LFW(
        args.train_set, transforms=transform, NUM_PER_CLS=20)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              args.test_batch_size,
                                              shuffle=False,
                                              num_workers=2)

    # ----------------------可视化训练数据
    def imshow(img, title=None):
        """Imshow for Tensor."""
        # (channels,imagesize,imagesize) -> (imagesize,imagesize,channels)
        img = img.numpy().transpose((1, 2, 0))  # 将Tensor中的数据格式转换用于plt显示的格式
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0.0, 1.0)

        plt.imshow(img)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

    # for i in range(4):  # 总共迭代4个batch的数据
    #     inputs, classes = next(iter(train_loader))  # 迭代一个batch的训练数据集
    #     out = torchvision.utils.make_grid(
    #         inputs[2])  # 每个batch有4个数据，每个数据包含3张图片的数据
    #     imshow(out)
    # ---------------------------------------------

    # ----------------------训练&测试
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    triplet_loss = nn.TripletMarginLoss(margin=1.2, p=2)  # 优化margin?

    # 优化器
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=1e-5)  # 权值衰减: 加入L2正则?

    def train(epoch):
        model.train()  # 网络在train模式
        total_loss = 0.0
        total_size = 0
        for batch_idx, (data, target) in enumerate(train_loader):  # 一个batch
            if args.cuda:
                data[0], target[0] = data[0].cuda(), target[0].cuda()
                data[1], target[1] = data[1].cuda(), target[1].cuda()
                data[2], target[2] = data[2].cuda(), target[2].cuda()

            data[0], target[0] = Variable(data[0]), Variable(target[0])
            data[1], target[1] = Variable(data[1]), Variable(target[1])
            data[2], target[2] = Variable(data[2]), Variable(target[2])

            optimizer.zero_grad()

            # 计算特征向量
            anchor = model.forward(data[0])
            positive = model.forward(data[1])
            negative = model.forward(data[2])

            # 计算分类loss
            loss_cls_0 = criterion(anchor, target[0].long())
            loss_cls_1 = criterion(positive, target[1].long())
            loss_cls_2 = criterion(negative, target[2].long())
            loss_cls = loss_cls_0 + loss_cls_1 + loss_cls_2

            # 计算三元组loss
            loss_tri = triplet_loss.forward(anchor, positive, negative)

            # 分类loss + triplet loss: 权重如何分配?
            loss = loss_tri + loss_cls

            # 统计loss
            total_loss += loss.data.cpu()[0]
            total_size += data[0].size(0)

            loss.backward()
            optimizer.step()

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)], Average loss: {:.4f}'.format(
                    epoch, batch_idx * len(data[0]), len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader), total_loss / total_size))

        if args.is_resume:
            model_path = args.check_path + \
                os.path.sep + 'epoch_{}.pth'.format(epoch)
            if os.path.exists(model_path):
                # 如果已经存在, 重命名模型
                print('the model already exists, rename the model and save.')
                ID = (epoch + 1) + LATEST_MODEL_ID
                print('new_id: ', ID)
                model_path = args.check_path + os.path.sep + \
                    'epoch_{}.pth'.format(ID)
            torch.save(model, model_path)
        else:
            if epoch % 10 == 0:
                model_path = args.check_path + \
                    os.path.sep + 'epoch_{}.pth'.format(epoch)
                if os.path.exists(model_path):
                    # 如果已经存在, 重命名模型
                    print('the model already exists, rename the model and save.')
                    ID = epoch + LATEST_MODEL_ID
                    model_path = args.check_path + os.path.sep + \
                        'epoch_{}.pth'.format(ID)
                torch.save(model, model_path)
        print('model {} saved.'.format(model_path))

    def test():
        model.eval()  # 网络在求值模式
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)

            output = model.forward(data)  # 预测

            test_loss += criterion(output, target).data.cpu()[0]
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / float(len(test_loader.dataset))))

    for epoch in range(args.epochs):
        train(epoch)
        test()
        validate(args.check_path)

# ----------------------验证数据集
def validate(check_path):
    if not os.path.exists(check_path):
        print('Error: invalid checkpoints path.')
        return
    print('Checkpoint path: ', check_path)

    # 加载网络
    checkpoints = os.listdir(check_path)
    checkpoints.sort(key=lambda x: int(re.match(r'epoch_(\d+)\.pth', x).group(1)),
                     reverse=True)
    model_path = os.path.join(check_path + os.path.sep + checkpoints[0])
    print('model: {}'.format(model_path))

    model = torch.load(model_path)
    model.eval()  # 网络在求值模式

    # 数据处理方式
    transform = transforms.Compose([
        # transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=np.array([0.485, 0.456, 0.406]),
            std=np.array([0.229, 0.224, 0.225])),
    ])

    # 加载数据
    valid_set = dataset.FACE_LFW('validate_set',
                                 transforms=transform,
                                 NUM_PER_CLS=10)
    valid_loader = torch.utils.data.DataLoader(valid_set,
                                               4,
                                               shuffle=False,
                                               num_workers=2)
    criterion = nn.CrossEntropyLoss()

    valid_loss = 0.0
    correct = 0
    is_cuda = torch.cuda.is_available()
    for data, target in tqdm(valid_loader):
        if is_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        output = model.forward(data)  # 预测

        valid_loss += criterion(output, target).data.cpu()[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    valid_loss /= float(len(valid_loader.dataset))
    print('Valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        valid_loss, correct, len(valid_loader.dataset),
        100. * correct / float(len(valid_loader.dataset))))


if __name__ == '__main__':
    test_triplet()
    # validate('checkpoints')
    # validate('resnet_checkpoints')
    # validate_statics('checkpoints')


# https://github.com/adambielski/siamese-triplet (pytorch triplet loss)
# https://www.ddvip.com/weixin/20171218A0236200.html (pytorch显存占用分析)
# https://blog.csdn.net/qq_14845119/article/details/76083042 (车型分类博客)
