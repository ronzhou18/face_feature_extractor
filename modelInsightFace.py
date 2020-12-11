# -*- coding: utf-8 -*-
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout2d, Dropout, AvgPool2d, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, Parameter,init
import torch.nn.functional as F
import torch
from collections import namedtuple
import math
import pdb

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''

def get_block(in_channel, depth, num_units, stride = 2):
  return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units-1)]

def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units = 3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    return blocks


class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        self.stride = stride
        if stride == 2:
            self.shortcut_layer = Sequential(Conv2d(in_channel, depth, (1, 1), stride ,bias=False),
                                             BatchNorm2d(depth,eps=2e-5,momentum=0.9))

            self.res_layer = Sequential(
                BatchNorm2d(in_channel,eps=2e-5,momentum=0.9),
                Conv2d(in_channel, depth, (3, 3), (1, 1), 1 ,bias=False),BatchNorm2d(depth,eps=2e-5,momentum=0.9),
                PReLU(depth),Conv2d(depth, depth, (3, 3), stride, 1 ,bias=False), BatchNorm2d(depth,eps=2e-5,momentum=0.9))
        else:

            self.res_layer = Sequential(
                BatchNorm2d(in_channel,eps=2e-5,momentum=0.9),
                Conv2d(in_channel, depth, (3, 3), (1, 1), 1 ,bias=False),BatchNorm2d(depth,eps=2e-5,momentum=0.9),
                PReLU(depth), Conv2d(depth, depth, (3, 3), stride, 1 ,bias=False), BatchNorm2d(depth,eps=2e-5,momentum=0.9))

    def forward(self, x):
        if self.stride == 2:
            shortcut = self.shortcut_layer(x)
            res = self.res_layer(x)
            return res + shortcut
        else:
            res = self.res_layer(x)
            return res+x




class Backbone(Module):
    def __init__(self, num_layers=50, drop_ratio=0.4,mode='ir',embedding_dim=256 ):
        super(Backbone, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR


        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64, eps=2e-5, momentum=0.9),
                                      PReLU(64))
        self.output_layer = Sequential(BatchNorm2d(512, eps=2e-5, momentum=0.9),
                                       Dropout(drop_ratio),
                                       Flatten(),
                                       Linear(512 * 7 * 7, embedding_dim),
                                       BatchNorm1d(embedding_dim, eps=2e-5, momentum=0.9,affine=True))
        modules = []
        item = 0
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))

        self.body = Sequential(*modules)

    def forward(self, x):

        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return l2_norm(x)

