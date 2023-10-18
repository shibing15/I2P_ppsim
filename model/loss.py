from numpy import positive
import torch
import torch.nn.functional as F
import numpy as np


def log_sigmoid(x):
    # for x > 0: 0 - log(1 + exp(-x))
    # for x < 0: x - log(1 + exp(x))
    # for x = 0: 0 (extra term for gradient stability)
    return torch.clamp(x, max=0) - torch.log(1 + torch.exp(-torch.abs(x))) + \
        0.5 * torch.clamp(x, min=0, max=0)

def log_minus_sigmoid(x):
    # for x > 0: -x - log(1 + exp(-x))
    # for x < 0:  0 - log(1 + exp(x))
    # for x = 0: 0 (extra term for gradient stability)
    return torch.clamp(-x, max=0) - torch.log(1 + torch.exp(-torch.abs(x))) + \
        0.5 * torch.clamp(x, min=0, max=0)

def siam_loss(img_features, pc_features, label, norm_factor):

    corr_map = torch.sum(img_features.unsqueeze(-1)*pc_features.unsqueeze(-2),dim=1) #(B, M, N)
    pos_mask = (label == 1)
    neg_mask = (label == 0)
    pos_num = pos_mask.sum().float()
    neg_num = neg_mask.sum().float()
    weight = label.new_zeros(label.size())
    weight[pos_mask] = 1 / pos_num
    weight[neg_mask] = 1 / neg_num * norm_factor
    weight /= weight.sum()
    return F.binary_cross_entropy_with_logits(corr_map, label, weight, reduction='sum')


def siam_loss2(img_features, pc_features, target, gamma):

    corr_map = torch.sum(img_features.unsqueeze(-1)*pc_features.unsqueeze(-2),dim=1)
    pos_mask = (target == 1)
    neg_mask = (target == 0)
    pos_log_sig = log_sigmoid(corr_map)
    neg_log_sig = log_minus_sigmoid(corr_map)

    prob = torch.sigmoid(corr_map)
    # pos_weight = torch.pow(1 - prob, gamma)
    pos_weight = torch.pow(1 - prob, gamma)
    neg_weight = torch.pow(prob, gamma)

    loss = -(target * pos_weight * pos_log_sig + \
        (1 - target) * neg_weight * neg_log_sig)
    
    avg_weight = target * pos_weight + (1 - target) * neg_weight
    loss /= avg_weight.mean()

    return loss.mean()

def neg_siam_loss2(img_features, pc_features, target, gamma):

    corr_map = torch.sum(img_features.unsqueeze(-1)*pc_features.unsqueeze(-2),dim=1)
    pos_mask = (target == 1)
    neg_mask = (target == 0)
    pos_log_sig = log_sigmoid(corr_map)
    neg_log_sig = log_minus_sigmoid(corr_map)

    prob = torch.sigmoid(corr_map)
    # pos_weight = torch.pow(1 - prob, gamma)
    pos_weight = torch.pow(prob, gamma)
    neg_weight = torch.pow(1- prob, gamma)

    loss = -(target * pos_weight * pos_log_sig + \
        (1 - target) * neg_weight * neg_log_sig)
    
    avg_weight = target * pos_weight + (1 - target) * neg_weight
    loss /= avg_weight.mean()

    return loss.mean()


#-----------------------------------------------------------------------------------#
def desc_loss(img_features,pc_features,mask,pos_margin=0.1,neg_margin=1.4,log_scale=10,num_kpt=512):
    pos_mask=mask#inline点中重投影误差小于阈值的点
    neg_mask=1-mask#inline点中重投影误差大于阈值的点
    #dists=torch.sqrt(torch.sum((img_features.unsqueeze(-1)-pc_features.unsqueeze(-2))**2,dim=1))
    #特征向量内积后计算cosine相似度，1-cosine相似度=cosine距离值域[0,2],越小表示描述子性能越好
    #最后得到的距离是计算图像特征与点云特征之间两两对应的距离或者说计算每个图像特征与每个点云特征的cosine distance
    dists=1-torch.sum(img_features.unsqueeze(-1)*pc_features.unsqueeze(-2),dim=1)#(B,N,N)
    #对inline点中重投影误差小于阈值的点计算权重，大的点权重置为0
    pos=dists-1e5*neg_mask
    #pos_margin的意思是距离小于pos_margin的点不用，认为这个点不需要被优化了？
    pos_weight=(pos-pos_margin).detach()
    pos_weight=torch.max(torch.zeros_like(pos_weight),pos_weight)
    #pos_weight[pos_weight>0]=1.
    #positive_row=torch.sum((pos[:,:num_kpt,:]-pos_margin)*pos_weight[:,:num_kpt,:],dim=-1)/(torch.sum(pos_weight[:,:num_kpt,:],dim=-1)+1e-8)
    #positive_col=torch.sum((pos[:,:,:num_kpt]-pos_margin)*pos_weight[:,:,:num_kpt],dim=-2)/(torch.sum(pos_weight[:,:,:num_kpt],dim=-2)+1e-8)
    #row行表示对于每个图像特征与筛选后的点云的距离
    lse_positive_row=torch.logsumexp(log_scale*(pos-pos_margin)*pos_weight,dim=-1)#(B,N)
    #col列表示对于每个点云特征与筛选后的图像之间的距离
    lse_positive_col=torch.logsumexp(log_scale*(pos-pos_margin)*pos_weight,dim=-2)#(B,N)
    #感觉是在体现选出的图像点与点云点的重要性？

    #将inline点中重投影误差大的neg点选出来，小的pos点加一个大偏移
    neg=dists+1e5*pos_mask
    #不考虑neg点中度量距离大于1.4的点，区分度足够不需要被优化？
    neg_weight=(neg_margin-neg).detach()
    neg_weight=torch.max(torch.zeros_like(neg_weight),neg_weight)
    #neg_weight[neg_weight>0]=1.
    #negative_row=torch.sum((neg[:,:num_kpt,:]-neg_margin)*neg_weight[:,:num_kpt,:],dim=-1)/torch.sum(neg_weight[:,:num_kpt,:],dim=-1)
    #negative_col=torch.sum((neg[:,:,:num_kpt]-neg_margin)*neg_weight[:,:,:num_kpt],dim=-2)/torch.sum(neg_weight[:,:,:num_kpt],dim=-2)
    lse_negative_row=torch.logsumexp(log_scale*(neg_margin-neg)*neg_weight,dim=-1)
    lse_negative_col=torch.logsumexp(log_scale*(neg_margin-neg)*neg_weight,dim=-2)

    '''
    softplus(x) = log(1+e^x)
    总体的意思是对于Pos点希望这些点计算得到的cosine distance尽量小于pos_margin
    对于neg点希望这些点这些点计算得到的cosine distance尽量大于neg_margin
    '''
    loss_col=F.softplus(lse_positive_row+lse_negative_row)/log_scale
    loss_row=F.softplus(lse_positive_col+lse_negative_col)/log_scale
    loss=loss_col+loss_row
    
    return torch.mean(loss),dists

def det_loss(img_score_inline,img_score_outline,pc_score_inline,pc_score_outline,dists,mask):
    #score (B,N)
    pids=torch.FloatTensor(np.arange(mask.size(-1))).to(mask.device)
    diag_mask=torch.eq(torch.unsqueeze(pids,dim=1),torch.unsqueeze(pids,dim=0)).unsqueeze(0).expand(mask.size()).float()
    furthest_positive,_=torch.max(dists*diag_mask,dim=1)     #(B,N)
    closest_negative,_=torch.min(dists+1e5*mask,dim=1)  #(B,N)
    loss_inline=torch.mean((furthest_positive-closest_negative)*(img_score_inline.squeeze()+pc_score_inline.squeeze()))
    loss_outline=torch.mean(img_score_outline)+torch.mean(pc_score_outline)
    return loss_inline+loss_outline

def det_loss2(img_score_inline,img_score_outline,pc_score_inline,pc_score_outline):
    #score (B,N)
    # pids=torch.FloatTensor(np.arange(mask.size(-1))).to(mask.device)
    # diag_mask=torch.eq(torch.unsqueeze(pids,dim=1),torch.unsqueeze(pids,dim=0)).unsqueeze(0).expand(mask.size()).float()
    # furthest_positive,_=torch.max(dists*diag_mask,dim=1)     #(B,N)
    # closest_negative,_=torch.min(dists+1e5*mask,dim=1)  #(B,N)
    #loss_inline=torch.mean((furthest_positive-closest_negative)*(img_score_inline.squeeze()+pc_score_inline.squeeze())) +torch.mean(1-img_score_inline)+torch.mean(1-pc_score_inline)
    loss_inline=torch.mean(1-img_score_inline)+torch.mean(1-pc_score_inline)
    loss_outline=torch.mean(img_score_outline)+torch.mean(pc_score_outline)
    return loss_inline+loss_outline


def cal_acc(img_features,pc_features,mask):
    dist=torch.sum((img_features.unsqueeze(-1)-pc_features.unsqueeze(-2))**2,dim=1) #(B,N,N)
    furthest_positive,_=torch.max(dist*mask,dim=1)
    closest_negative,_=torch.min(dist+1e5*mask,dim=1)
    '''print(furthest_positive)
    print(closest_negative)
    print(torch.max(torch.sum(mask,dim=1)))
    assert False'''
    diff=furthest_positive-closest_negative
    accuracy=(diff<0).sum(dim=1)/dist.size(1)
    return accuracy