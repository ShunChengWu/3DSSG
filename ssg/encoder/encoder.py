from os import sys
if __name__ == '__main__' and __package__ is None:
    sys.path.append('../../../')
    sys.path.append('../../../detr')
else:
    sys.path.append('detr')
import torch
import torch.nn as nn
import SSG2D.define
from SSG2D.utils.util import normalize_imagenet, read_label_files
from detr.hubconf import detr_resnet101_panoptic, detr_resnet50_dc5_panoptic, detr_resnet50_panoptic

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)
def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(out_bbox.device)
    return b

class DETR(nn.Module):
    r''' DETR
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    '''

    def __init__(self, normalize=True, threshold=0.85, **Args):
        super().__init__()
        self.normalize = normalize
        self.threshold = threshold
        # self.model, self.postprocessor = torch.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', pretrained=True, return_postprocessor=True, num_classes=250)
        self.model, self.postprocessor = detr_resnet101_panoptic(pretrained=True, 
                                               return_postprocessor=True, 
                                               num_classes=250,
                                               threshold=threshold)
        self.model.eval();
        
    def forward(self, x):        
        if self.normalize:
            x = normalize_imagenet(x)
        out = self.model(x)
        result = self.postprocessor(out, torch.as_tensor(x.shape[-2:]).unsqueeze(0))[0]
        return result, out
    
    def show(self,x):
        import io,numpy
        from PIL import Image
        from panopticapi.utils import rgb2id
        import itertools
        import seaborn as sns
        # import matplotlib.pyplot as plt
        result, out = self(x)
        palette = itertools.cycle(sns.color_palette())
        # The segmentation is stored in a special-format png
        panoptic_seg = Image.open(io.BytesIO(result['png_string']))
        panoptic_seg = numpy.array(panoptic_seg, dtype=numpy.uint8).copy()
        # We retrieve the ids corresponding to each mask
        panoptic_seg_id = rgb2id(panoptic_seg)
        
        # Finally we color each mask individually
        panoptic_seg[:, :, :] = 0
        # print('debug: panoptic_seg_id.max():',panoptic_seg_id.max())
        for id in range(panoptic_seg_id.max() + 1):
          panoptic_seg[panoptic_seg_id == id] = numpy.asarray(next(palette)) * 255
          
        ''' draw object detection '''
        probas = out['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > self.threshold
        bboxes_scaled = rescale_bboxes(out['pred_boxes'][0, keep], (x.shape[-1],x.shape[-2]))
        
        im = Image.fromarray(panoptic_seg, 'RGB')
        self.plot_results(im, probas[keep], bboxes_scaled)
          
        return torch.tensor(numpy.array(panoptic_seg)).to(x.device)
    
    def plot_results(self,pil_img, prob, boxes):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(16,10))
        plt.imshow(pil_img)
        ax = plt.gca()
        classes = read_label_files(SSG2D.define.PATH_LABEL_COCOSTUFF, ': ')
        # colors = COLORS * 100
        # for segment_info in prob['segments_info']:
        #     idx = segment_info['id']
        #     cl = segment_info['category_id']
        #     xmin, ymin, xmax, ymax = boxes[idx]
        #     ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
        #                                 fill=False, color=torch.rand(3).tolist(), linewidth=3))
        #     text = f'{classes[cl]}' if cl < len(classes) else '-'#': {p[cl]:0.2f}'
        #     ax.text(xmin, ymin, text, fontsize=15,
        #     bbox=dict(facecolor='yellow', alpha=0.5))
        for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                        fill=False, color=torch.rand(3).tolist(), linewidth=3))
            cl = p.argmax()
            # text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
            text = '.'
            text = f'{classes[cl.item()]}' if cl < len(classes) else '-'#': {p[cl]:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))
        plt.axis('off')
        plt.show()
       