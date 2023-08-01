from .point_classifier import PointNetCls, PointNetRelClsMulti, PointNetRelCls
from . import image_classifier
import torch.nn as nn


def get_cvr(in_channels, out_channels):
    return nn.Sequential(
        nn.Linear(in_channels, 256),
        nn.Dropout(),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(256, out_channels)
    )


def get_res18(in_channels, out_channels):
    return nn.Linear(in_channels, out_channels)


class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FastRCNNPredictor, self).__init__()
        assert in_channels is not None
        num_inputs = in_channels
        num_classes = out_channels
        self.cls_score = nn.Linear(num_inputs, num_classes)
        # self.cls_score.weight = torch.nn.init.xavier_normal(self.cls_score.weight, gain=1.0)
        nn.init.normal_(self.cls_score.weight, mean=0, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        cls_logit = self.cls_score(x)
        return cls_logit


classifier_list = {
    "pointnet": PointNetCls,
    "pointnet_multi": PointNetRelClsMulti,
    "pointnet_rel": PointNetRelCls,
    "vgg16": image_classifier.VGG16,
    # "basic": image_classifier.basic,
    'basic': get_res18,
    'cvr': get_cvr,
    'res18': get_res18,
    'imp': FastRCNNPredictor,
}
