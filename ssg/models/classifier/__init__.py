from .point_classifier import PointNetCls, PointNetRelClsMulti, PointNetRelCls
from . import image_classifier
import torch.nn as nn
def get_cvr(in_channels, out_channels):
    return nn.Sequential(
                nn.Linear(in_channels, 256),
                nn.Dropout(),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256,out_channels)
            )
def get_res18(in_channels, out_channels):
    return nn.Linear(in_channels, out_channels)

classifider_list = {
	"pointnet": PointNetCls,
	"pointnet_multi": PointNetRelClsMulti,
	"pointnet_rel": PointNetRelCls,
	"vgg16": image_classifier.VGG16,
    # "basic": image_classifier.basic,
    'basic': get_res18,
    'cvr': get_cvr,
    'res18': get_res18,
}
