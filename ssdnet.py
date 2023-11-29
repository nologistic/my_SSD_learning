import time
import torch
from torch import nn, optim
import sys
import torch.nn.init as init
import os

sys.path.append("../..")
import d2lzh_pytorch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''
L2Norm层，在conv4_3特征图大小为38*38，512通道,此时方差可能较大，需要进行标准化
不能直接标准化，因为会改变层规模，而且减慢学习速度，故采用放缩系数来来标准化
'''


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels  # 输入通道
        self.gamma = scale or None  # 放缩系数
        self.eps = 1e-10  # 尾部的一个调整参数，防止除0
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        # x /= norm
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class SSDnet(nn.Module):#num_classes, backbone, pretrained
    def __init__(self, num_classes, size, phase):
        super(SSDnet, self).__init__()
        self.phase = phase  # 是否预训练
        self.num_classes = num_classes  # 分类数目，我们只用来检测人脸，设为1
        self.L2Norm = L2Norm(512, 20)
        mbox = [4, 6, 6, 6, 4, 4]  # 各个特征图的先验框数量
        loc_layers = []  # 特征图之后的卷积层
        conf_layers = []

        loc_layers += [nn.Conv2d(512, mbox[0] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(512, mbox[0] * num_classes, kernel_size=3, padding=1)]

        loc_layers += [nn.Conv2d(1024, mbox[1] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(1024, mbox[1] * num_classes, kernel_size=3, padding=1)]

        loc_layers += [nn.Conv2d(512, mbox[2] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(512, mbox[2] * num_classes, kernel_size=3, padding=1)]

        loc_layers += [nn.Conv2d(256, mbox[3] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, mbox[3] * num_classes, kernel_size=3, padding=1)]

        loc_layers += [nn.Conv2d(256, mbox[4] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, mbox[4] * num_classes, kernel_size=3, padding=1)]

        loc_layers += [nn.Conv2d(256, mbox[5] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, mbox[5] * num_classes, kernel_size=3, padding=1)]

        self.loc = nn.ModuleList(loc_layers)
        self.conf = nn.ModuleList(conf_layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # 输入三通道，输出64通道，卷积核3*3，填充1步
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 2*2最大池化,步长为2

            nn.Conv2d(64, 128, 3, 1, 1),  # 输入64通道，输出128通道，卷积核3*3，填充1步
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),  # 输in128通道，输出128通道，卷积核3*3，填充1步
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, 1),  # in_channels 128 out_channels 256 kernel_size 3*3  padding 1 stride 1
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),  # in 256 , out 256 kernel 3*3 padding 1 stride 1
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),  # in 256 , out 256 kernel 3*3 padding 1 stride 1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, ceil_mode=True),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        # 输出第一预测特征层
        self.conv2 = nn.Sequential(

            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 1, 1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        # 输出第二预测特征层
        # 第一附加层
        # 在resource保留特征图时，是对未激活的特征图进行保留，故relu层放到前面
        self.conv3 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
        )
        # 输出第三预测特征层
        # 第二附加层
        self.conv4 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 128,  kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        )
        # 输出第三预测特征层
        # 第三附加层
        self.conv5 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256,kernel_size= 3,stride= 1)
        )
        # 输出第三预测特征层
        # 第四附加层
        self.conv6 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3,stride= 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()
        x = self.conv1(x)
        L1 = self.L2Norm(x)  # 对conv4_3的结果标准化后并保存
        sources.append(L1)  # 保存4——3
        # 获得conv4_3第一特征层的输出
        x = self.conv2(x)
        # 获得conv7，第二特征层的输出
        sources.append(x)
        x = self.conv3(x)
        sources.append(x)
        # 获得conv8_2第三特征层的输出
        x = self.conv4(x)
        sources.append(x)
        # 获得conv9_2,第4特征层的输出
        x = self.conv5(x)
        sources.append(x)
        # 获得conv10_2 第5特征层的输出
        x = self.conv6(x)
        sources.append(x)
        # 获得conv11_2 第6特征层的输出




        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())  # 保存位置信息
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())  # 保存置信度
        # 将loc中每个张量进行展开为二维，并以第一维度进行拼接
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        # 将loc中每个张量进行展开为二维，并以第一维度进行拼接
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        #   loc会reshape到batch_size, num_anchors, 4
        #   conf会reshap到batch_size, num_anchors, self.num_classes
        output = (
            loc.view(loc.size(0), -1, 4),
            conf.view(conf.size(0), -1, self.num_classes),
        )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('装在模型参数...')
            self.load_state_dict(torch.load(base_file,
                                            map_location=lambda storage, loc: storage))
            print('完成!')
        else:
            print('参数文件错误')

if __name__ == "__main__":
    net = SSDnet(phase='trian',size=300,num_classes=20)
    for i, layer in enumerate(net.conv1):
        print(i, layer)
    for i, layer in enumerate(net.conv2):
        print(i, layer)
    for i, layer in enumerate(net.conv3):
        print(i, layer)
    for i, layer in enumerate(net.conv4):
        print(i, layer)
    for i, layer in enumerate(net.conv5):
        print(i, layer)
    for i, layer in enumerate(net.conv6):
        print(i, layer)
    # for i,layer in enumerate(net.loc):
    #     print(i,layer)
    # for i,layer in enumerate(net.conf):
    #     print(i,layer)
