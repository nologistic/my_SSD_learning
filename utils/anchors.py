import numpy as np


class AnchorBox():
    def __init__(self, input_shape, min_size, max_size=None, aspect_ratios=None, flip=True):
        self.input_shape = input_shape#输入尺寸
        self.min_size = min_size#最小形状
        self.max_size = max_size#最大形状
        self.aspect_ratios = []#长款比
        for ar in aspect_ratios:
            self.aspect_ratios.append(ar)
            self.aspect_ratios.append(1.0 / ar)
    def call(self,layer_shape,mask=None):
        layer_height    = layer_shape[0]
        layer_width     = layer_shape[1]
        ##获得输入特征层的宽高
        img_height  = self.input_shape[0]
        img_width   = self.input_shape[1]
        #获取输入进来的图片的宽和高
        box_widths  = []
        box_heights = []
        for ar in self.aspect_ratios:
            #添加一个小正方形
            if ar==1 and len(box_widths)==0:
                box_widths.append(self.min_size)
                box_heights.append(self.min_size)
            elif ar == 1 and len(box_widths) > 0:
                #添加大正方形
                box_widths.append(np.sqrt(self.min_size * self.max_size))
                box_heights.append(np.sqrt(self.min_size * self.max_size))
            # 然后添加长方形
            elif ar != 1:
                box_widths.append(self.min_size * np.sqrt(ar))
                box_heights.append(self.min_size / np.sqrt(ar))
        #获取所有先验框的一半长宽
        box_widths  = 0.5 * np.array(box_widths)
        box_heights = 0.5 * np.array(box_heights)
        #每个特征层对于的步长
        step_x = img_width / layer_width
        step_y = img_height / layer_height
        #生成每个特征层的网格中心
        linx = np.linspace(0.5 * step_x, img_width - 0.5 * step_x,
                           layer_width)
        liny = np.linspace(0.5 * step_y, img_height - 0.5 * step_y,
                           layer_height)
        centers_x, centers_y = np.meshgrid(linx, liny)
        centers_x = centers_x.reshape(-1, 1)
        centers_y = centers_y.reshape(-1, 1)
        # 每一个先验框需要两个(centers_x, centers_y)，前一个用来计算左上角，后一个计算右下角
        num_anchors_ = len(self.aspect_ratios)
        anchor_boxes = np.concatenate((centers_x, centers_y), axis=1)
        anchor_boxes = np.tile(anchor_boxes, (1, 2 * num_anchors_))
        # 获得先验框的左上角和右下角
        anchor_boxes[:, ::4]    -= box_widths
        anchor_boxes[:, 1::4]   -= box_heights
        anchor_boxes[:, 2::4]   += box_widths
        anchor_boxes[:, 3::4]   += box_heights
        anchor_boxes[:, ::2]    /= img_width
        anchor_boxes[:, 1::2]   /= img_height
        anchor_boxes = anchor_boxes.reshape(-1, 4)
        #整合成二维张量
        anchor_boxes = np.minimum(np.maximum(anchor_boxes, 0.0), 1.0)
        #保证范围
        return anchor_boxes

def get_vgg_output_length(height, width):
    filter_sizes    = [3, 3, 3, 3, 3, 3, 3, 3]
    padding         = [1, 1, 1, 1, 1, 1, 0, 0]
    stride          = [2, 2, 2, 2, 2, 2, 1, 1]
    feature_heights = []
    feature_widths  = []

    for i in range(len(filter_sizes)):
        height  = (height + 2*padding[i] - filter_sizes[i]) // stride[i] + 1
        width   = (width + 2*padding[i] - filter_sizes[i]) // stride[i] + 1
        feature_heights.append(height)
        feature_widths.append(width)
    return np.array(feature_heights)[-6:], np.array(feature_widths)[-6:]


def get_anchors(input_shape=[300, 300], anchors_size=[30, 60, 111, 162, 213, 264, 315], backbone='vgg'):

    feature_heights, feature_widths = get_vgg_output_length(input_shape[0], input_shape[1])
    aspect_ratios = [[1, 2], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2], [1, 2]]
    anchors = []
    for i in range(len(feature_heights)):
        anchor_boxes = AnchorBox(input_shape, anchors_size[i], max_size=anchors_size[i + 1],
                                 aspect_ratios=aspect_ratios[i]).call([feature_heights[i], feature_widths[i]])
        anchors.append(anchor_boxes)
    anchors = np.concatenate(anchors, axis=0)
    return anchors
