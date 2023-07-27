import torch
import torch.nn as nn
from ssg.models.node_encoder.base import NodeEncoderBase
# from torchvision import models
from torchvision.ops import roi_align
import logging
logger_py = logging.getLogger(__name__)


class ROI_EXTRACTOR(NodeEncoderBase):
    def __init__(self, cfg, backbone: str, device):
        '''
        Takes input images and proposeal. Return node and edge features
        '''
        super().__init__(cfg, backbone, device)
        # self.global_pooling_method = cfg.model.image_encoder.aggr

    def reset_parameters(self):
        pass

    def forward(self, images, proposals, edge_indices):
        '''get image features'''
        images = self.preprocess(images)
        width, height = images.shape[-1], images.shape[-2]

        proposals[:, 1] *= width
        proposals[:, 2] *= height
        proposals[:, 3] *= width
        proposals[:, 4] *= height

        '''build union bounding boxes'''
        union_boxes = list()
        if len(edge_indices):
            if edge_indices.shape[1] != 2:
                edge_indices = edge_indices.t()
            assert edge_indices.shape[1] == 2
            for ind_pair in edge_indices:
                pairwise_boxes = proposals[ind_pair]
                assert pairwise_boxes[0, 0] == pairwise_boxes[1, 0]
                # union
                x_min = pairwise_boxes[:, 1].min()
                y_min = pairwise_boxes[:, 2].min()
                x_max = pairwise_boxes[:, 3].max()
                y_max = pairwise_boxes[:, 4].max()
                union_boxes.append(
                    [pairwise_boxes[0, 0], x_min, y_min, x_max, y_max])
        union_boxes = torch.FloatTensor(union_boxes).to(images.device)

        if self.use_global:
            node_features = roi_align(images, proposals, self.roi_region)
            node_features = node_features.view(
                node_features.shape[0], -1)  # [views, cdim]
            node_features = self.nn_post(node_features)

            if len(union_boxes) > 0:
                edge_features = roi_align(images, union_boxes, self.roi_region)
                edge_features = edge_features.view(
                    edge_features.shape[0], -1)  # [views, cdim]
                edge_features = self.nn_post(edge_features)
            else:
                edge_features = torch.FloatTensor().to(images.device)
        else:
            node_features = torch.zeros(
                [proposals.shape[0], self.node_feature_dim], device=self._device)
            for i in range(proposals.shape[0]):
                w = (proposals[i, 3] - proposals[i, 1]).item()
                h = (proposals[i, 4] - proposals[i, 2]).item()
                w, h = int(w), int(h)
                x = roi_align(images, proposals[i].unsqueeze(0), [h, w])
                x = self.nn_enc(x)
                x = self.nn_post(x).flatten()
                node_features[i] = x

            edge_features = torch.zeros(
                [union_boxes.shape[0], self.node_feature_dim], device=self._device)
            for i in range(union_boxes.shape[0]):
                w = (union_boxes[i, 3] - union_boxes[i, 1]).item()
                h = (union_boxes[i, 4] - union_boxes[i, 2]).item()
                w, h = int(w), int(h)
                x = roi_align(images, union_boxes[i].unsqueeze(0), [h, w])
                x = self.nn_enc(x)
                x = self.nn_post(x).flatten()
                edge_features[i] = x

        return node_features, edge_features


if __name__ == '__main__':
    import codeLib
    config = codeLib.Config('./experiments/config_IMP_full_l20_0.yaml')
    # config.path = '/home/sc/research/PersistentSLAM/python/2DTSG/data/test/'

    model = ROI_EXTRACTOR(config, backbone='res18', device='cpu')
    if config.data.use_precompute_img_feature:
        images = torch.rand([3, 512, 32, 32])
    else:
        images = torch.rand([3, 3, 512, 512])
    bboxes = torch.FloatTensor([
        [0, 0, 0, 1, 1],
        [0, 0.5, 0.5, 0.7, 0.7],
        [1, 0, 0, 0.5, 0.5],
        [1, 0, 0, 1, 1]])

    edge_indices = [
        [0, 1],
        [1, 0],
        [2, 3],
        [3, 2]
    ]
    edge_indices = torch.tensor(edge_indices, dtype=torch.long)

    output = model(images, proposals=bboxes, edge_indices=edge_indices)
