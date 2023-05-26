from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#from model.utils.config import cfg
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.faster_rcnn_uda.faster_rcnn_uda import _fasterRCNN
from model.faster_rcnn_uda.discriminator import netD_pixel,netD_mid,netD,netD_da
from model.faster_rcnn_uda.randomlayer import RandomLayer
from model.faster_rcnn_uda.Resnet import resnet101, resnet50



class resnet(_fasterRCNN):
  def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False,lc=False,gc=False, la_attention = False
               ,mid_attention = False):
    self.dout_base_model = 1024
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic
    self.lc = lc
    self.gc = gc
    self.layers = num_layers
    if self.layers == 101:
      self.model_path = './data/pretrained_model/resnet101_caffe.pth'
    if self.layers == 50:
      self.model_path = './data/pretrained_model/resnet50_caffe.pth'
    if self.layers == 152:
      self.model_path = './data/pretrained_model/resnet152_caffe.pth'
    if self.layers == 18:
      self.model_path = './data/pretrained_model/resnet18_caffe.pth'


    _fasterRCNN.__init__(self, classes, class_agnostic,lc,gc, la_attention, mid_attention)

  def _init_modules(self):

    resnet = resnet101()
    if self.layers == 50:
      resnet = resnet50()
    if self.pretrained == True:
      print("Loading pretrained weights from %s" %(self.model_path))
      state_dict = torch.load(self.model_path)
      resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})
    # Build resnet.
    self.RCNN_base1 = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
      resnet.maxpool,resnet.layer1)
    self.RCNN_base2 = nn.Sequential(resnet.layer2)
    self.RCNN_base3 = nn.Sequential(resnet.layer3)

    self.netD_pixel = netD_pixel(context=self.lc)
    self.netD = netD(context=self.gc)
    self.netD_mid = netD_mid(context=self.gc)

    self.RCNN_top = nn.Sequential(resnet.layer4)
    feat_d = 2048
    feat_d2 = 384
    feat_d3 = 1024

    self.RandomLayer = RandomLayer([feat_d, feat_d2], feat_d3)
    self.RandomLayer.cuda()

    self.netD_da = netD_da(feat_d3)

    self.RCNN_cls_score = nn.Linear(feat_d+feat_d2, self.n_classes)
    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(feat_d+feat_d2, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(feat_d+feat_d2, 4 * self.n_classes)

    # Fix blocks
    for p in self.RCNN_base1[0].parameters(): p.requires_grad=False
    for p in self.RCNN_base1[1].parameters(): p.requires_grad=False


    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

    self.RCNN_base1.apply(set_bn_fix)
    self.RCNN_base2.apply(set_bn_fix)
    self.RCNN_base3.apply(set_bn_fix)
    self.RCNN_top.apply(set_bn_fix)

  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    nn.Module.train(self, mode)
    if mode:
      # Set fixed blocks to be in eval mode
      self.RCNN_base1.eval()
      self.RCNN_base1[4].train()
      self.RCNN_base2.train()
      self.RCNN_base3.train()

      def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
          m.eval()

      self.RCNN_base1.apply(set_bn_eval)
      self.RCNN_base2.apply(set_bn_eval)
      self.RCNN_base3.apply(set_bn_eval)
      self.RCNN_top.apply(set_bn_eval)

  def _head_to_tail(self, pool5):
    fc7 = self.RCNN_top(pool5).mean(3).mean(2)
    return fc7
