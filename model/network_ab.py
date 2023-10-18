import torch
import torch.nn as nn
import torch.nn.functional as F
from . import layers_pc
from . import imagenet
from .imagenet import ResidualConv,ImageUpSample
from . import pointnet2
from .siamese_net import SiameseNet
from data.options import Options

class SiamI2P_ab(nn.Module):
    def __init__(self,opt:Options):
        super(SiamI2P_ab, self).__init__()
        self.opt=opt
        self.pc_encoder=pointnet2.PCEncoder(opt,Ca=64,Cb=256,Cg=512)
        self.img_encoder=imagenet.ImageEncoder()

        self.H_fine_res = int(round(self.opt.img_H / self.opt.img_fine_resolution_scale))
        self.W_fine_res = int(round(self.opt.img_W / self.opt.img_fine_resolution_scale))

        self.node_b_attention_pn = layers_pc.PointNet(256+512,
                                               [256, self.H_fine_res*self.W_fine_res],
                                               activation=self.opt.activation,
                                               normalization=self.opt.normalization,
                                               norm_momentum=opt.norm_momentum,
                                               norm_act_at_last=False)
        self.node_b_pn = layers_pc.PointNet(256+512+512+512,
                                            [1024, 512, 512],
                                            activation=self.opt.activation,
                                            normalization=self.opt.normalization,
                                            norm_momentum=opt.norm_momentum,
                                            norm_act_at_last=False)
                                            
        self.node_a_attention_pn = layers_pc.PointNet(64 + 512,
                                                      [256, int(self.H_fine_res * self.W_fine_res * 4)],
                                                      activation=self.opt.activation,
                                                      normalization=self.opt.normalization,
                                                      norm_momentum=opt.norm_momentum,
                                                      norm_act_at_last=False)

        self.node_a_pn = layers_pc.PointNet(64+256+512,
                                            [512, 128, 128],
                                            activation=self.opt.activation,
                                            normalization=self.opt.normalization,
                                            norm_momentum=opt.norm_momentum,
                                            norm_act_at_last=False)

        per_point_pn_in_channels = 32 + 64 + 128 + 512
        self.per_point_pn=layers_pc.PointNet(per_point_pn_in_channels,
                                            [256, 256, 128],
                                            activation=self.opt.activation,
                                            normalization=self.opt.normalization,
                                            norm_momentum=opt.norm_momentum,
                                            norm_act_at_last=True,
                                                )

        self.pc_feature_layer=nn.Sequential(nn.Conv1d(128,128,1,bias=False),nn.BatchNorm1d(128),nn.ReLU(),nn.Conv1d(128,128,1,bias=False),nn.BatchNorm1d(128),nn.ReLU(),nn.Conv1d(128,64,1,bias=False),nn.BatchNorm1d(64))
        self.pc_score_layer=nn.Sequential(nn.Conv1d(128,128,1,bias=False),nn.BatchNorm1d(128),nn.ReLU(),nn.Conv1d(128,64,1,bias=False),nn.BatchNorm1d(64),nn.ReLU(),nn.Conv1d(64,1,1,bias=False),nn.Sigmoid())
    
        #self.img_32_attention_conv=nn.Sequential(ResidualConv(512+512,512,kernel_1=True),ResidualConv(512,512,kernel_1=True),ResidualConv(512,self.opt.node_b_num,kernel_1=True))
        #self.img_16_attention_conv=nn.Sequential(ResidualConv(512+256,256,kernel_1=True),ResidualConv(256,256,kernel_1=True),ResidualConv(256,self.opt.node_a_num,kernel_1=True))
        self.img_32_attention_conv=nn.Sequential(   nn.Conv2d(512+512,512,1,bias=False),nn.BatchNorm2d(512),nn.ReLU(),
                                                    nn.Conv2d(512,512,1,bias=False),nn.BatchNorm2d(512),nn.ReLU(),
                                                    nn.Conv2d(512,self.opt.node_b_num,1,bias=False))
        self.img_16_attention_conv=nn.Sequential(   nn.Conv2d(512+256,256,1,bias=False),nn.BatchNorm2d(256),nn.ReLU(),
                                                    nn.Conv2d(256,256,1,bias=False),nn.BatchNorm2d(256),nn.ReLU(),
                                                    nn.Conv2d(256,self.opt.node_a_num,1,bias=False))


        self.up_conv1=ImageUpSample(768+320,256)
        self.up_conv2=ImageUpSample(256+128,128)
        self.up_conv3=ImageUpSample(128+64+64,64)

        self.img_feature_layer=nn.Sequential(nn.Conv2d(64,64,1,bias=False),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,64,1,bias=False),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,64,1,bias=False), nn.BatchNorm2d(64))
        self.img_score_layer=nn.Sequential(nn.Conv2d(64,64,1,bias=False),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,64,1,bias=False),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,1,1,bias=False),nn.Sigmoid())

        # self.siamese_layer = SiameseNet(64,
        #                                 [128, 256,512,128, 64])
        self.siamese_layer = SiameseNet(64,
                                    [128, 256, 128, 64])

    def gather_topk_features(self, min_k_idx, features):
        """

        :param min_k_idx: BxNxk
        :param features: BxCxM
        :return:
        """
        B, N, k = min_k_idx.size(0), min_k_idx.size(1), min_k_idx.size(2)
        C, M = features.size(1), features.size(2)

        return torch.gather(features.unsqueeze(3).expand(B, C, M, k),
                            index=min_k_idx.unsqueeze(1).expand(B, C, N, k),
                            dim=2)  # BxCxNxk

    def upsample_by_interpolation(self,
                                  interp_ab_topk_idx,
                                  node_a,
                                  node_b,
                                  up_node_b_features):
        interp_ab_topk_node_b = self.gather_topk_features(interp_ab_topk_idx, node_b)  # Bx3xMaxk
        # Bx3xMa -> Bx3xMaxk -> BxMaxk
        interp_ab_node_diff = torch.norm(node_a.unsqueeze(3) - interp_ab_topk_node_b, dim=1, p=2, keepdim=False)
        interp_ab_weight = 1 - interp_ab_node_diff / torch.sum(interp_ab_node_diff, dim=2, keepdim=True)  # BxMaxk
        interp_ab_topk_node_b_features = self.gather_topk_features(interp_ab_topk_idx, up_node_b_features)  # BxCxMaxk
        # BxCxMaxk -> BxCxMa
        interp_ab_weighted_node_b_features = torch.sum(interp_ab_weight.unsqueeze(1) * interp_ab_topk_node_b_features,
                                                       dim=3)
        return interp_ab_weighted_node_b_features
    def forward(self,pc,intensity,sn,img,node_a,node_b):
        #node_a=FPS(pc,self.opt.node_a_num)
        #node_b=FPS(pc,self.opt.node_b_num)
        B,N,Ma,Mb=pc.size(0),pc.size(2),node_a.size(2),node_b.size(2)

        pc_center,\
        cluster_mean, \
        node_a_min_k_idx, \
        first_pn_out, \
        second_pn_out, \
        node_a_features, \
        node_b_features, \
        global_feature = self.pc_encoder(pc,
                                          intensity,
                                          sn,
                                          node_a,
                                          node_b)

        '''print(node_a_features.size())
        print(node_b_features.size())'''
        
        #print(global_feature.size())

        C_global = global_feature.size(1)

        img_feature_set=self.img_encoder(img)

        '''for i in img_feature_set:
            print(i.size())'''


        img_global_feature=img_feature_set[-1]  #512
        img_s32_feature_map=img_feature_set[-2] #512
        img_s16_feature_map=img_feature_set[-3] #256
        img_s8_feature_map=img_feature_set[-4]  #128
        img_s4_feature_map=img_feature_set[-5]  #64
        img_s2_feature_map=img_feature_set[-6]  #64

        img_feature_ab = img_s4_feature_map.flatten(start_dim=2)
        
        
        return pc_center,\
            cluster_mean, \
            node_a_min_k_idx, \
            first_pn_out, \
            second_pn_out, \
            node_a_features, \
            node_b_features, \
            global_feature, \
            img_global_feature, \
            img_s32_feature_map, \
            img_s16_feature_map, \
            img_s8_feature_map, \
            img_s4_feature_map, \
            img_s2_feature_map, \
            img_feature_ab

if __name__=='__main__':
    opt=Options()
    pc=torch.rand(10,3,20480).cuda()
    intensity=torch.rand(10,1,20480).cuda()
    sn=torch.rand(10,3,20480).cuda()
    img=torch.rand(10,3,160,512).cuda()
    net=SiamI2P_ab(opt).cuda()

    pc_center,\
    cluster_mean, \
    node_a_min_k_idx, \
    first_pn_out, \
    second_pn_out, \
    node_a_features, \
    node_b_features, \
    global_feature, \
    img_global_feature, \
    img_s32_feature_map, \
    img_s16_feature_map, \
    img_s8_feature_map, \
    img_s4_feature_map, \
    img_s2_feature_map=net(pc,intensity,sn,img)
    import ipdb; ipdb.set_trace

    