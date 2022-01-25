import torch
from torch import nn
from codeLib.utils.util import pytorch_count_params
import ssg2d
import logging
import torchvision
logger_py = logging.getLogger(__name__)


class Res18Enc(nn.Module):
    def __init__(self, num_obj_cls, freeze:bool):
        super().__init__()
        resnet18=torchvision.models.resnet18(pretrained=True)
        nn_enc = resnet18
        nn_enc.fc = nn.Identity()
        if freeze:
            nn_enc.eval()
            for param in nn_enc.parameters(): param.requires_grad = False
        self.model = nn_enc
    def forward(self,x):
        return {'nodes_feature': self.model(x)}

class VGG16Enc(nn.Module):
    def __init__(self,num_obj_cls, freeze:bool):
        super().__init__()
        vgg16=torchvision.models.vgg16(pretrained=True)
        if freeze:
            nn_enc = vgg16
            nn_enc.classifier=nn.Identity()
            nn_enc.eval()
            for param in nn_enc.parameters(): param.requires_grad = False
        self.model = nn_enc
    def forward(self,x):
        return {'nodes_feature': self.model(x)}
        

class SVEnc(nn.Module):
    def __init__(self,cfg, num_obj_cls, device):
        super().__init__()
        self.cfg=cfg
        self._device=device
        self.method = cfg.model.node_encoder.method
        models = dict()
        
        if self.method == 'inceptionv4':
            logger_py.info('use GVCNN original implementation')
            encoder = ssg2d.models.node_encoder.GVCNN.SVCNN(num_obj_cls)
            # c = ssg2d.models.encoder.inceptionV4.inceptionv4()
            # encoder.last_linear = nn.Linear(1536, num_obj_cls)
            # encoder.last_linear.apply(init_weights)
            classifier = nn.Identity()
        elif self.method == 'res18':
            logger_py.info('use Res18')
            encoder = Res18Enc(num_obj_cls,True)
            classifier = nn.Linear(512, num_obj_cls)
        elif self.method == 'vgg16':
            logger_py.info('use_vgg16')
            encoder = VGG16Enc(num_obj_cls,True)
            classifier = ssg2d.models.classifier.classifider_list['vgg16'](25088,num_obj_cls)
        else:
            raise NotImplementedError()
            
        '''print modified config'''
        if cfg.VERBOSE: print(cfg)  
        '''set models'''
        models['encoder'] = encoder
        models['classifier'] = classifier
        params = list()
        print('==trainable parameters==')
        for name, model in models.items():
            if model is None: 
                self.name = model
                continue
            if len(cfg.GPU) > 1:
                model = torch.nn.DataParallel(model, cfg.GPU)
            model = model.to(device)
            self.add_module(name, model)
            params += list(model.parameters())
            print(name,pytorch_count_params(model))
        print('')
    def forward(self,**args):
        embeds = self.encoder(**args)        
        node_cls = self.classifier(embeds['nodes_feature'])
        output=embeds
        output['node_cls'] = node_cls
        return output
    
    def compute_loss(self):
        pass
    
    def calculate_metrics(self,**args):
        node_cls_pred = args['node_cls_pred']
        node_cls_pred = torch.softmax(node_cls_pred, dim=1)
        node_cls_gt   = args['node_cls_gt']
        node_cls_pred = torch.max(node_cls_pred,1)[1]
        acc_node_cls = (node_cls_gt == node_cls_pred).sum().item() / node_cls_gt.nelement()
        return {
            'acc_node_cls': acc_node_cls
        }
    
    
if __name__ == '__main__':
    import codeLib
    config = codeLib.Config('./configs/default_mv.yaml')
    config.path = '/home/sc/research/PersistentSLAM/python/2DTSG/data/test/'
    # config.model.node_encoder.method='mvcnn'
    config.model.node_encoder.method='inceptionv4'
    # config.model.node_encoder.backend='res18'
    model = SVEnc(config,num_obj_cls=40,device='cpu')
    print(model)
    # if config.data.use_precompute_img_feature:
    #     images = torch.rand([3,512,32,32])
    # else:
    #     images = torch.rand([3,3,512,512])
    # bboxes = [
    #     {0: torch.FloatTensor([0,0,1,1]), 1: torch.FloatTensor([0,0,0.5,0.5])},
    #     {1: torch.FloatTensor([0,0,1,1])},
    #     ]
    # output = model(images,bboxes)