import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


from model.utils.config import cfg

from model.rpn.rpn import _RPN
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from torchvision.ops import roi_align as ROIAlign
from torchvision.ops import roi_pool as ROIPool



from model.faster_rcnn_uda.discriminator import ImageDA
from model.faster_rcnn_uda.discriminator import InstanceDA

from model.utils.net_utils import _smooth_l1_loss, grad_reverse, local_attention, middle_attention

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic,lc,gc, la_attention = False, mid_attention = False):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0
        self.lc = lc
        self.gc = gc
        self.la_attention = la_attention
        self.mid_attention = mid_attention

        # Define RPN
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)


        self.RCNN_roi_pool = ROIPool
        self.RCNN_roi_align = ROIAlign

        # Declaration loss adaptation        
        self.RCNN_imageDA = ImageDA(self.dout_base_model)
        self.RCNN_instanceDA = InstanceDA()
        self.consistency_loss = torch.nn.MSELoss(size_average=False)


    def forward(self, im_data, im_info, gt_boxes, num_boxes, target=False, eta=1.0):

        if self.training:
            if target:
                self.RCNN_rpn.eval()
            else:
                self.RCNN_rpn.train()
        
        batch_size = im_data.size(0)
        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        

        # Feed image data to first base model to obtain first base feature map
        base_feat1 = self.RCNN_base1(im_data)
        if self.lc:
            d_pixel, _ = self.netD_pixel(grad_reverse(base_feat1, eta))
            _, feat_pixel = self.netD_pixel(base_feat1.detach())
        else:
            d_pixel = self.netD_pixel(grad_reverse(base_feat1, eta))

        if self.la_attention:
            base_feat1 = local_attention(base_feat1, d_pixel.detach())


	# Feed first base feature map to second base model to obtain  midlle base feature map
        base_feat2 = self.RCNN_base2(base_feat1)
        if self.gc:
            domain_mid, _ = self.netD_mid(grad_reverse(base_feat2, eta))
            _, feat_mid = self.netD_mid(base_feat2.detach())
        else:
            domain_mid = self.netD_mid(grad_reverse(base_feat2, eta))

        if self.mid_attention:
            base_feat2 = middle_attention(base_feat2, domain_mid.detach())

	
	# Feed midlle base feature map to third base model to obtain  global base feature map
        base_feat = self.RCNN_base3(base_feat2)
        if self.gc:
            domain_p, _ = self.netD(grad_reverse(base_feat, eta))
            _,feat = self.netD(base_feat.detach())
        else:
            domain_p = self.netD(grad_reverse(base_feat, eta))
        
        base_score = self.RCNN_imageDA(grad_reverse(base_feat, 0.1))
        
        # feed base feature map tp RPN to obtain rois
        #if target==True:
        #    self.RCNN_rpn.train()
        #else:
        #    self.RCNN_rpn.eval()
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)
        
        
        # if it is training phrase, then use ground truth bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        
        # Do ROI pooling based on predicted rois
        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5),(cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)
        else:
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5),(cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0)

        # Feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)
        
        instance_sigmoid = self.RCNN_instanceDA(grad_reverse(pooled_feat, 0.1))
        #print(same_size_label.size())
        #print(instance_sigmoid.size())


        #feat_pixel = torch.zeros(feat_pixel.size()).cuda()
        #Instance Feature Calculate
        feat_pixel = feat_pixel.view(1, -1).repeat(pooled_feat.size(0), 1)
        feat_mid = feat_mid.view(1, -1).repeat(pooled_feat.size(0), 1)
        feat = feat.view(1, -1).repeat(pooled_feat.size(0), 1)
        # concat
        feat = torch.cat((feat_mid, feat), 1)
        feat = torch.cat((feat_pixel, feat), 1)
        #
        feat_random = self.RandomLayer([pooled_feat, feat])

        d_ins = self.netD_da(grad_reverse(feat_random, eta))

        if target:
            return d_pixel, domain_p, domain_mid, d_ins, base_score,instance_sigmoid 

        pooled_feat = torch.cat((feat, pooled_feat), 1)
        
	# Compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)

        if self.training and not self.class_agnostic:
	    # Select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # Compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, \
		d_pixel, domain_p,domain_mid, d_ins, base_score, instance_sigmoid 

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
