import torch
from torch import nn
from codeLib.utils.util import pytorch_count_params
from ssg2d.models.node_encoder import node_encoder_list
from ssg2d.models.classifier import classifider_list
import logging
logger_py = logging.getLogger(__name__)

class MVEnc(nn.Module):
    def __init__(self,cfg, num_obj_cls, device):
        super().__init__()
        logger_py.setLevel(cfg.log_level)
        self.cfg=cfg
        self._device=device
        self.method = cfg.model.node_encoder.method
        models = dict()
        
        if self.method == 'cvr':
            logger_py.info('use CVR original implementation')
            cfg.model.node_encoder.backend = 'res18'
            cfg.model.node_feature_dim=512
            encoder = node_encoder_list[self.method](cfg,cfg.model.node_encoder.backend,device)
            node_feature_dim = encoder.node_feature_dim
            # classifier = ssg2d.models.classifider_list['basic'](in_channels=node_feature_dim, out_channels=num_obj_cls)
            classifier = classifider_list['cvr'](in_channels=512, out_channels=num_obj_cls)
        elif self.method == 'mvcnn':
            logger_py.info('use MVCNN original implementation')
            cfg.model.node_encoder.backend = 'vgg16'
            cfg.model.node_feature_dim=25088
            cfg.model.node_encoder.aggr = 'max'
            encoder = node_encoder_list[self.method](cfg,cfg.model.node_encoder.backend,device)
            node_feature_dim = encoder.node_feature_dim
            classifier = classifider_list['vgg16'](in_channels=node_feature_dim, out_channels=num_obj_cls)
        elif self.method == 'gvcnn':
            logger_py.info('use GVCNN original implementation')
            encoder =  node_encoder_list[self.method](cfg,num_obj_cls,device)
            classifier = torch.nn.Identity()
        elif self.method == 'rnn':# use RNN with the assumption of Markov Random Fields(MRFs) and CRF  https://www.robots.ox.ac.uk/~szheng/papers/CRFasRNN.pdf
            raise NotImplementedError()
        elif self.method == 'mean':# prediction by running mean.
            logger_py.info('use mean cls feature')
            encoder = node_encoder_list[self.method](cfg,num_obj_cls,cfg.model.node_encoder.backend,device)
            classifier = torch.nn.Identity()
            # raise NotImplementedError()
        elif self.method == 'gmu':# graph memory unit
            logger_py.info('use GMU')
            encoder = node_encoder_list[self.method](cfg,num_obj_cls,cfg.model.node_encoder.backend,device)
            if cfg.model.mean_cls:
                classifier = torch.nn.Identity()
            else:
                node_feature_dim = cfg.model.gmu.memory_dim
                node_clsifier = "res18" if cfg.model.node_classifier.method == 'basic' else cfg.model.node_classifier.method #default is res18
                classifier = classifider_list[node_clsifier](in_channels=node_feature_dim, out_channels=num_obj_cls)
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
    config.model.node_encoder.method='mvcnn'
    config.model.node_encoder.method='mean'
    config.model.node_encoder.backend='res18'
    model = MVEnc(config,num_obj_cls=40,device='cpu')
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