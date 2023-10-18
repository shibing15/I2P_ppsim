import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time

from . import operations
from .layers_pc import *
from typing import Tuple, List

class SiameseNet(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels_list: List[int],
                 normalization: str='batch',
                 norm_momentum: float=0.1,
                 activation: str='relu',
                 output_init_radius: float=None,
                 norm_act_at_last: bool=False,
                 dropout_list: List[float]=None ):
        super(SiameseNet, self).__init__()
        self.feature_net = PointNet(in_channels,
                                    out_channels_list,
                                    normalization=normalization,
                                    norm_momentum=norm_momentum,
                                    norm_act_at_last=False)
        
    def forward(self,  x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        '''
        Tensor x and y should have the same feature dim
        '''
        # print(x.shape, y.shape)
        assert x.shape[1] == y.shape[1]
        x = self.feature_net(x)
        y = self.feature_net(y)

        return x, y

def get_dataset_add(file_list :list):
    address = []
    for file in file_list:
        filename = file.split('.')[0]
        filename_nonum = filename.split('_')
        if filename_nonum[-1] == 'norm':
            filename_nonum.pop(-2)
        else:
            filename_nonum.pop(-1)
        fileadd = '_'.join(filename_nonum)
        address.append(fileadd)

    address = list(set(address))

    return address

import os

if __name__ == '__main__':
    
    #set data path
    root_path = '/home/ai-i-sunyunda/code/CorrI2P/kitti_train_result_all_dist_thres_1.00_pos_margin_0.20_neg_margin_1.80'
    file_list = os.listdir(root_path)
    address = get_dataset_add(file_list)

    model = SiameseNet(64,[128, 64, 32]).cuda()

    
    