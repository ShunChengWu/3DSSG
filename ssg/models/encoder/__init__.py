from . import image_encoders
from . import point_encoders

image_encoder_dict = {
    'simple_conv': image_encoders.ConvEncoder,
    'resnet18': image_encoders.Resnet18,
    'resnet34': image_encoders.Resnet34,
    'resnet50': image_encoders.Resnet50,
    'resnet101': image_encoders.Resnet101,
    'vgg16': image_encoders.Vgg16,
}

point_encoder_dict = {
    'pointnet': point_encoders.PointNetfeat
}
