import copy
from tqdm import tqdm
from collections import defaultdict
from codeLib.models import BaseTrainer
from codeLib.common import check_weights, convert_torch_to_scalar
import torch
from ssg.utils.util_eva import EvalSceneGraphBatch
import time
import logging
import codeLib.utils.moving_average as moving_average
import ssg
from ssg import define
from ssg.trainer.eval_inst import EvalInst
from ssg.utils.util_eva import merged_prediction_to_node
logger_py = logging.getLogger(__name__)


class Trainer_IMP(BaseTrainer, EvalInst):
    def __init__(self, cfg, model, node_cls_names: list, edge_cls_names: list,
                 device=None,  **kwargs):
        super().__init__(device)
        logger_py.setLevel(cfg.log_level)
        self.cfg = cfg
        self.model = model  # .to(self._device)
        self.w_node_cls = kwargs.get('w_node_cls', None)
        self.w_edge_cls = kwargs.get('w_edge_cls', None)
        self.node_cls_names = node_cls_names  # kwargs['node_cls_names']
        self.edge_cls_names = edge_cls_names  # kwargs['edge_cls_names']

        trainable_params = filter(
            lambda p: p.requires_grad, model.parameters())
        self.optimizer = ssg.config.get_optimizer(cfg, trainable_params)

        if self.w_node_cls is not None:
            logger_py.info('train with weighted node class.')
            self.w_node_cls = self.w_node_cls.to(self._device)
        if self.w_edge_cls is not None:
            logger_py.info('train with weighted node class.')
            self.w_edge_cls = self.w_edge_cls.to(self._device)

        self.eva_tool = EvalSceneGraphBatch(self.node_cls_names, self.edge_cls_names, multi_rel_prediction=self.cfg.model.multi_rel,
                                            k=0, none_name=define.NAME_NONE)  # do not calculate topK in training mode
        self.loss_node_cls = torch.nn.CrossEntropyLoss(weight=self.w_node_cls)
        if self.cfg.model.multi_rel:
            self.loss_rel_cls = torch.nn.BCEWithLogitsLoss(
                pos_weight=self.w_edge_cls)
        else:
            self.loss_rel_cls = torch.nn.CrossEntropyLoss(
                weight=self.w_edge_cls)

    def zero_metrics(self):
        self.eva_tool.reset()

    def evaluate(self, val_loader, topk):
        it_dataset = val_loader.__iter__()
        eval_tool = EvalSceneGraphBatch(self.node_cls_names, self.edge_cls_names, multi_rel_prediction=self.cfg.model.multi_rel, k=topk, save_prediction=True,
                                        none_name=define.NAME_NONE)
        eval_list = defaultdict(moving_average.MA)

        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for data in tqdm(it_dataset, leave=False):
            eval_step_dict = self.eval_step(data, eval_tool=eval_tool)

            for k, v in eval_step_dict.items():
                eval_list[k].update(v)
            # break
        eval_dict = dict()
        obj_, edge_ = eval_tool.get_mean_metrics()
        for k, v in obj_.items():
            # print(k)
            eval_dict[k+'_node_cls'] = v
        for k, v in edge_.items():
            # print(k)
            eval_dict[k+'_edge_cls'] = v

        for k, v in eval_list.items():
            eval_dict[k] = v.avg

        vis = self.visualize(eval_tool=eval_tool)
        eval_dict['visualization'] = vis
        del it_dataset
        return eval_dict, eval_tool

    def sample(self, dataloader):
        pass

    def train_step(self, data, it=None):
        self.model.train()
        self.optimizer.zero_grad()
        logs = self.compute_loss(data, it=it, eval_tool=self.eva_tool)
        if 'loss' not in logs:
            return logs
        logs['loss'].backward()
        check_weights(self.model.state_dict())
        self.optimizer.step()
        return logs

    def eval_step(self, data, eval_tool=None):
        ''' Performs a validation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()
        eval_dict = {}
        with torch.no_grad():
            eval_dict = self.compute_loss(
                data, eval_mode=True, eval_tool=eval_tool)
        eval_dict = convert_torch_to_scalar(eval_dict)
        # for (k, v) in eval_dict.items():
        #     eval_dict[k] = v.item()
        return eval_dict

    def compute_loss(self, data, eval_mode=False, it=None, eval_tool=None):
        ''' Compute the loss.

        Args:
            data (dict): data dictionary
            eval_mode (bool): whether to use eval mode
            it (int): training iteration
        '''
        # Initialize loss dictionary and other values
        logs = {}

        # Process data dictionary
        data.to(self._device)
        # data = self.process_data_dict(data)

        # Shortcuts
        # scan_id = data['scan_id']
        gt_cls = data['roi'].y
        gt_rel = data['roi', 'to', 'roi'].y
        # mask2instance = data['roi'].idx2oid[0]
        # edge_index = data['roi','to','roi'].edge_index
        # gt_relationships = data['relationships']
        # gt_cls = data['image_gt_cls']
        # gt_rel = data['image_gt_rel']
        # mask2instance = data['image_mask2instance']
        # node_edges_ori = data['image_node_edges']
        # data['image_node_edges'] = data['image_node_edges'].t().contiguous()
        # if 'temporal_node_graph' in data: data['temporal_node_graph']=data['temporal_node_graph'].t().contiguous()
        # if 'temporal_edge_graph' in data: data['temporal_edge_graph']=data['temporal_edge_graph'].t().contiguous()

        # check input valid
        # if node_edges_ori.ndim==1: return {}

        # print('')
        # print('gt_rel.sum():',gt_rel.sum())

        ''' make forward pass through the network '''
        node_cls, edge_cls = self.model(data)

        ''' calculate loss '''
        logs['loss'] = 0

        if self.cfg.training.lambda_mode == 'dynamic':
            # calculate loss ratio base on the number of node and edge
            batch_node = node_cls.shape[0]
            self.cfg.training.lambda_node = 1
            if edge_cls is not None:
                batch_edge = edge_cls.shape[0]
                self.cfg.training.lambda_edge = batch_edge / batch_node

        ''' 1. node class loss'''
        self.calc_node_loss(logs, node_cls, gt_cls, self.w_node_cls)

        ''' 2. edge class loss '''
        self.calc_edge_loss(logs, edge_cls, gt_rel, self.w_edge_cls)

        '''3. get metrics'''
        metrics = self.model.calculate_metrics(
            node_cls_pred=node_cls,
            node_cls_gt=gt_cls,
            edge_cls_pred=edge_cls,
            edge_cls_gt=gt_rel
        )
        for k, v in metrics.items():
            logs[k] = v

        ''' eval tool '''
        if eval_tool is not None:
            node_cls = torch.softmax(node_cls.detach(), dim=1)
            data['roi'].pd = node_cls.detach()
            if edge_cls is not None:
                edge_cls = torch.sigmoid(edge_cls.detach())
                data['roi', 'to', 'roi'].pd = edge_cls.detach()
            # TODO: merge roi to node?
            merged_prediction_to_node(data)

            eval_tool.add(data)
        return logs
        # return loss if eval_mode else loss['loss']

    def calc_node_loss(self, logs, node_cls_pred, node_cls_gt, weights=None):
        '''
        calculate node loss.
        can include
        classification loss
        attribute loss
        affordance loss
        '''
        # loss_obj = F.nll_loss(node_cls_pred, node_cls_gt, weight = weights)
        loss_obj = self.loss_node_cls(node_cls_pred, node_cls_gt)
        logs['loss'] += self.cfg.training.lambda_node * loss_obj
        logs['loss_obj'] = loss_obj

    def calc_edge_loss(self, logs, edge_cls_pred, edge_cls_gt, weights=None):
        if len(edge_cls_gt) == 0:
            logs['loss'] += 0  # self.cfg.training.lambda_edge * loss_rel
            logs['loss_rel'] = 0
            return
        if self.cfg.model.multi_rel:
            loss_rel = self.loss_rel_cls(edge_cls_pred, edge_cls_gt)
        else:
            loss_rel = self.loss_rel_cls(edge_cls_pred, edge_cls_gt)

        logs['loss'] += self.cfg.training.lambda_edge * loss_rel
        logs['loss_rel'] = loss_rel

    def visualize(self, eval_tool=None):
        if eval_tool is None:
            eval_tool = self.eva_tool
        node_confusion_matrix, edge_confusion_matrix = eval_tool.draw(
            plot_text=False,
            grid=False,
            normalize='log',
            plot=False
        )
        return {
            'node_confusion_matrix': node_confusion_matrix,
            'edge_confusion_matrix': edge_confusion_matrix
        }

    def toDevice(self, *args):
        output = list()
        for item in args:
            if isinstance(item,  torch.Tensor):
                output.append(item.to(self._device))
            elif isinstance(item,  dict):
                ks = item.keys()
                vs = self.toDevice(*item.values())
                item = dict(zip(ks, vs))
                output.append(item)
            elif isinstance(item, list):
                output.append(self.toDevice(*item))
            else:
                output.append(item)
        return output

    def get_log_metrics(self):
        output = dict()
        obj_, edge_ = self.eva_tool.get_mean_metrics()

        for k, v in obj_.items():
            output[k+'_node_cls'] = v
        for k, v in edge_.items():
            output[k+'_edge_cls'] = v
        return output
