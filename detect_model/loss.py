import torch
from torch.autograd import Variable


### 此处默认真实值和预测值的格式均为 bs * W * H * channels
import torch
import torch.nn as nn

def dice_coefficient(y_true_cls, y_pred_cls,
                     training_mask):
    '''
    dice loss
    :param y_true_cls:
    :param y_pred_cls:
    :param training_mask:
    :return:
    '''
    eps = 1e-5
    intersection =torch.sum(y_true_cls * y_pred_cls * training_mask)
    union = torch.sum(y_true_cls * training_mask) + torch.sum(y_pred_cls * training_mask) + eps
    loss = 1. - (2 * intersection / union)

    return loss

class LossFunc(nn.Module):
    def __init__(self):
        super(LossFunc, self).__init__()
        return 
    
    def forward(self, y_true_cls, y_pred_cls, y_true_geo, y_pred_geo, training_mask):
        classification_loss = dice_coefficient(y_true_cls, y_pred_cls, training_mask)
        # scale classification loss to match the iou loss part
        classification_loss *= 1

    #     # d1 -> top, d2->right, d3->bottom, d4->left
    # #     d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
    #     d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = torch.split(y_true_geo, 1, 1)
    # #     d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)
    #     d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = torch.split(y_pred_geo, 1, 1)
    #     area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
    #     area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
    #     w_union = torch.min(d2_gt, d2_pred) + torch.min(d4_gt, d4_pred)
    #     h_union = torch.min(d1_gt, d1_pred) + torch.min(d3_gt, d3_pred)
    #     area_intersect = w_union * h_union
    #     area_union = area_gt + area_pred - area_intersect
    #     L_AABB = -torch.log((area_intersect + 1.0)/(area_union + 1.0))
    #     L_theta = 1 - torch.cos(theta_pred - theta_gt)
    #     L_g = L_AABB + 20 * L_theta
        diff = y_pred_geo - y_true_geo
        diff /= 10

        abs_diff = torch.abs(diff)
        abs_diff_lt_1 = torch.less(abs_diff, 1)
        L_g = torch.where(abs_diff_lt_1, 0.5 * torch.square(abs_diff), abs_diff - 0.5)
        # L_g = L_g / gt_min_edge
        L_g = torch.mean(L_g, axis=1)

        y_true_cls = torch.squeeze(y_true_cls)
        training_mask = torch.squeeze(training_mask)

        geometry_loss = torch.mean(L_g * y_true_cls * training_mask)

        return geometry_loss + classification_loss

