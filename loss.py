import math
from functools import partial

import torch
import torch.nn as nn


class MultiboxLoss(nn.Module):
    def __init__(self, num_classes, alpha=1.0, neg_pos_ratio=3.0,
                 background_label_id=0, negatives_for_hard=100.0):
        self.num_classes = num_classes
        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratio
        if background_label_id != 0:
            raise Exception("0必须作为背景标签")
        self.background_label_id = background_label_id
        self.negatives_for_hard = torch.FloatTensor([negatives_for_hard])[0]

    def _l1_smooth_loss(self, y_true, y_pred):
        abs_loss = torch.abs(y_true - y_pred)
        sq_loss = 0.5 * (y_true - y_pred) ** 2
        l1_loss = torch.where(abs_loss < 1.0, sq_loss, abs_loss - 0.5)
        return torch.sum(l1_loss, -1)

    def _softmax_loss(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, min=1e-7)
        softmax_loss = -torch.sum(y_true * torch.log(y_pred), axis=-1)
        return softmax_loss

    def forward(self, y_true, y_pred):
        # --------------------------------------------- #
        #   y_true batch_size, 8732, 4  + 1(background class) + self.num_classes
        #   y_pred batch_size, 8732, 4 + self.num_classes
        #训练批次数目，预选框的数目，4代表位置参数，num_classes代表种类
        # --------------------------------------------- #
        num_boxes = y_true.size()[1]
        #h获得预选框数量
        y_pred = torch.cat([y_pred[0], nn.Softmax(-1)(y_pred[1])], dim=-1)
        #将预测结果softmax前和softmax后的结果保存在一起
        conf_loss = self._softmax_loss(y_true[:, :, 4:-1], y_pred[:, :, 4:])
        #计算分类损失，softmax损失
        loc_loss = self._l1_smooth_loss(y_true[:, :, :4], y_pred[:, :, :4])
        #计算位置回归损失，用smooth损失计算
        pos_loc_loss = torch.sum(loc_loss * y_true[:, :, -1],
                                 axis=1)
        #计算正样本总的位置回归loss
        pos_conf_loss = torch.sum(conf_loss * y_true[:, :, -1],
                                  axis=1)
        #计算正样本的总的分类回归loss
        num_pos=torch.sum(y_true[:,:,-1],axis=-1)
        #计算每张图正样本总数
        num_neg=torch.min(self.neg_pos_ratio*num_pos,num_boxes-num_pos)
        #每张图负样本总数有两个选择，一，正样本总数*neg_pos_ratio,二，预选框总数减去正样本数量
        pos_num_neg_mask=num_neg>0
        #选取负样本数量大于0的，生成布尔掩码
        has_min=torch.sum(pos_num_neg_mask)
        #计算负样本总数量
        num_neg_batch=torch.sum(num_neg) if has_min >0 else self.negatives_for_hard
        #如果负样本数量大于0则总批次负样本数量尾num_neg求和，否则为默认负样本数量100
        confs_start = 4 + self.background_label_id + 1
        #计算分类标签的索引头不包括背景
        confs_end   = confs_start + self.num_classes - 1
        #计算分类标签的索引尾
        max_confs = torch.sum(y_pred[:, :, confs_start:confs_end], dim=2)
        #把不是背景的概率求和，求和后的概率越大，代表越难分类
        max_confs   = (max_confs * (1 - y_true[:, :, -1])).view([-1])
        #将有正样本的框置为0，没有正样本的框置为1*max_confs，再展成一维向量
        _, indices = torch.topk(max_confs, k=int(num_neg_batch.cpu().numpy().tolist()))
        #取出前num_neg_batch个最难分类的负样本的索引
        neg_conf_loss = torch.gather(conf_loss.view([-1]), 0, indices)
        #取出最难分类负样本的分类损失总和
        num_pos     = torch.where(num_pos != 0, num_pos, torch.ones_like(num_pos))
        #如果每张图的正样本不为0，则保留，否则置为1
        total_loss  = torch.sum(pos_conf_loss) + torch.sum(neg_conf_loss) + torch.sum(self.alpha * pos_loc_loss)
        #计算总体损失
        total_loss  = total_loss / torch.sum(num_pos)
        #对总损失平均化
        return total_loss
def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

