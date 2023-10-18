import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import argparse
from model.network import SiamI2P
# from data.kitti_pc_img_dataloader import kitti_pc_img_dataset
# from nuscenes.nuscenes_pc_img_dataloader import nuScenesLoader
from oxford.siami2p_oxford_pc_img_dataloader import OxfordLoader
from model.loss import  det_loss2, siam_loss, siam_loss2, neg_siam_loss2
import numpy as np
import datetime
import logging
import math
from oxford import options
import cv2
from scipy.spatial.transform import Rotation


def get_P_diff(P_pred_np,P_gt_np):
    P_diff=np.dot(np.linalg.inv(P_pred_np),P_gt_np)
    t_diff=np.linalg.norm(P_diff[0:3,3])
    r_diff=P_diff[0:3,0:3]
    R_diff=Rotation.from_matrix(r_diff)
    angles_diff=np.sum(np.abs(R_diff.as_euler('xzy',degrees=True)))
    return t_diff,angles_diff


def test_acc(model,testdataloader,args):
    
    t_diff_set=[]
    angles_diff_set=[]
    t_val_diff_set = []
    angles_val_diff_set = []
    success_rate = 0
    for step,data in enumerate(testdataloader):
        if step%10==0:
            model.eval()
            img=data['img'].cuda()              #full size
            pc=data['pc'].cuda()
            intensity=data['intensity'].cuda()
            sn=data['sn'].cuda()
            K=data['K'].cuda()
            P=data['P'].cuda()
            pc_mask=data['pc_mask'].cuda()      
            img_mask=data['img_mask'].cuda()    #1/4 size

            pc_kpt_idx=data['pc_kpt_idx'].cuda()                #(B,512)
            pc_outline_idx=data['pc_outline_idx'].cuda()
            img_kpt_idx=data['img_kpt_idx'].cuda()
            img_outline_idx=data['img_outline_index'].cuda()
            node_a=data['node_a'].cuda()
            node_b=data['node_b'].cuda()

            img_siam_feature_norm, pc_siam_feature_norm, \
            img_score,pc_score=model(pc,intensity,sn,img,node_a,node_b)     #64 channels feature
            
            img_score=img_score[0].data.cpu().numpy()
            pc_score=pc_score[0].data.cpu().numpy()
            img_siam_feature=img_siam_feature_norm[0].data.cpu().numpy()
            pc_feature=pc_siam_feature_norm[0].data.cpu().numpy()
            pc=pc[0].data.cpu().numpy()
            P=P[0].data.cpu().numpy()
            K=K[0].data.cpu().numpy()
            
            img_x=np.linspace(0,np.shape(img_score)[-1]-1,np.shape(img_score)[-1]).reshape(1,-1).repeat(np.shape(img_score)[-2],0).reshape(1,np.shape(img_score)[-2],np.shape(img_score)[-1])
            img_y=np.linspace(0,np.shape(img_score)[-2]-1,np.shape(img_score)[-2]).reshape(-1,1).repeat(np.shape(img_score)[-1],1).reshape(1,np.shape(img_score)[-2],np.shape(img_score)[-1])

            img_xy=np.concatenate((img_x,img_y),axis=0)

            img_xy_flatten=img_xy.reshape(2,-1)
            # img_feature_flatten=img_feature.reshape(np.shape(img_feature)[0],-1)
            img_feature_flatten = img_siam_feature
            img_score_flatten=img_score.squeeze().reshape(-1)

            img_index=(img_score_flatten>args.img_thres)
            #topk_img_index=np.argsort(-img_score_flatten)[:args.num_kpt]
            img_xy_flatten_sel=img_xy_flatten[:,img_index]
            img_feature_flatten_sel=img_feature_flatten[:,img_index]
            img_score_flatten_sel=img_score_flatten[img_index]

            pc_index=(pc_score.squeeze()>args.pc_thres)
            #topk_pc_index=np.argsort(-pc_score.squeeze())[:args.num_kpt]
            pc_sel=pc[:,pc_index]
            pc_feature_sel=pc_feature[:,pc_index]
            pc_score_sel=pc_score.squeeze()[pc_index]

            #以点云为主
            # dist= np.sum(np.expand_dims(pc_feature_sel,axis=2)*np.expand_dims(img_feature_flatten_sel,axis=1),axis=0)
            # # sel_index=np.argmax(-dist,axis=1)
            # sel_index=np.argsort(-dist,axis=1)[:,0]
            # img_xy_pc=img_xy_flatten_sel[:,sel_index]

            #以图像数为主
            dist=np.sum(np.expand_dims(pc_feature_sel,axis=2)*np.expand_dims(img_feature_flatten_sel,axis=1),axis=0)
            sel_index=np.argsort(-dist,axis=0)[0,:]
            # import ipdb; ipdb.set_trace()
            # img_xy_pc=img_xy_flatten_sel[:,sel_index]
            img_xy_pc = img_xy_flatten_sel
            pc_sel = pc_sel[:, sel_index]

            is_success,R,t,inliers=cv2.solvePnPRansac(pc_sel.T,img_xy_pc.T,K,useExtrinsicGuess=False,
                                                        iterationsCount=500,
                                                        reprojectionError=args.dist_thres,
                                                        flags=cv2.SOLVEPNP_EPNP,
                                                        distCoeffs=None)
            T_pred=np.eye(4)
            if is_success:
                R,_=cv2.Rodrigues(R)
                T_pred[0:3,0:3]=R
                T_pred[0:3,3:]=t
            t_diff,angles_diff=get_P_diff(T_pred,P)
            if (t_diff < 5) & (angles_diff < 10):
                t_val_diff_set.append(t_diff)
                angles_val_diff_set.append(angles_diff)
            t_diff_set.append(t_diff)
            angles_diff_set.append(angles_diff)
    success_rate = len(t_val_diff_set) / len(t_diff_set)
    return np.mean(np.array(t_diff_set)),np.mean(np.array(angles_diff_set)), np.mean(np.array(t_val_diff_set)), np.mean(np.array(angles_val_diff_set)), success_rate

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--epoch', type=int, default=25, metavar='epoch',
                        help='number of epoch to train')
    parser.add_argument('--train_batch_size', type=int, default=16, metavar='train_batch_size',
                        help='Size of train batch')
    parser.add_argument('--val_batch_size', type=int, default=8, metavar='val_batch_size',
                        help='Size of val batch')
    parser.add_argument('--data_path', type=str, default='/media/ai-i-sunyunda/data/data/oxford_for_i2p/', metavar='data_path',
                        help='train and test data path')
    parser.add_argument('--num_point', type=int, default=20480, metavar='num_point',
                        help='point cloud size to train')
    parser.add_argument('--num_workers', type=int, default=8, metavar='num_workers',
                        help='num of CPUs')
    parser.add_argument('--val_freq', type=int, default=5000, metavar='val_freq',
                        help='')
    parser.add_argument('--lr', type=float, default=0.001, metavar='lr',
                        help='')
    parser.add_argument('--min_lr', type=float, default=0.00001, metavar='lr',
                        help='')

    parser.add_argument('--P_tx_amplitude', type=float, default=10, metavar='P_tx_amplitude',
                        help='')
    parser.add_argument('--P_ty_amplitude', type=float, default=0.5, metavar='P_ty_amplitude',
                        help='')
    parser.add_argument('--P_tz_amplitude', type=float, default=10, metavar='P_tz_amplitude',
                        help='')
    parser.add_argument('--P_Rx_amplitude', type=float, default=0.0 * math.pi / 12.0, metavar='P_Rx_amplitude',
                        help='')
    parser.add_argument('--P_Ry_amplitude', type=float, default=2*math.pi, metavar='P_Ry_amplitude',
                        help='')
    parser.add_argument('--P_Rz_amplitude', type=float, default=0.0 * math.pi / 12.0, metavar='P_Rz_amplitude',
                        help='')

    parser.add_argument('--save_path', type=str, default='./runs/oxofrd_log_xy_20480_128', metavar='save_path',
                        help='path to save log and model')
    '''parser.add_argument('--save_path', type=str, default='./only_test', metavar='save_path',
                        help='path to save log and model')'''
    
    parser.add_argument('--num_kpt', type=int, default=512, metavar='num_kpt',
                        help='')
    parser.add_argument('--dist_thres', type=float, default=1, metavar='num_kpt',
                        help='')

    parser.add_argument('--img_thres', type=float, default=0.9, metavar='img_thres',
                        help='')
    parser.add_argument('--pc_thres', type=float, default=0.9, metavar='pc_thres',
                        help='')

    parser.add_argument('--pos_margin', type=float, default=0.2, metavar='pos_margin',
                        help='')
    parser.add_argument('--neg_margin', type=float, default=1.8, metavar='neg_margin',
                        help='')


    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    logdir=os.path.join(args.save_path, 'dist_thres_%0.2f_pos_margin_%0.2f_neg_margin_%0.2f'%(args.dist_thres,args.pos_margin,args.neg_margin,))
    try:
        os.makedirs(logdir)
    except:
        print('mkdir failue')

    logger=logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/log.txt' % (logdir))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    opt=options.Options()
    model=SiamI2P(opt)

    train_dataset = OxfordLoader(args.data_path, 'train', opt=opt)
    test_dataset = OxfordLoader(args.data_path, 'val', opt=opt)
    assert len(train_dataset) > 10
    assert len(test_dataset) > 10
    trainloader=torch.utils.data.DataLoader(train_dataset,batch_size=args.train_batch_size,shuffle=True,drop_last=True,num_workers=args.num_workers)
    testloader=torch.utils.data.DataLoader(test_dataset,batch_size=args.val_batch_size,shuffle=False,drop_last=True,num_workers=args.num_workers)

    # checkpoints = torch.load('/home/ai-i-sunyunda/code/I2P_own/runs/nuscenes_log_xy_20480_128_newkitti/dist_thres_1.00_pos_margin_0.20_neg_margin_1.80/mode_epoch_21.t7')
    # model.load_state_dict(checkpoints)
    model=model.cuda()

    current_lr=args.lr
    learnable_params=filter(lambda p:p.requires_grad,model.parameters())
    optimizer=torch.optim.Adam(learnable_params,lr=current_lr)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch, eta_min=args.lr)
    logger.info(args)

    global_step=0

    best_t_diff=1000
    best_r_diff=1000

    for epoch in range(args.epoch):

        for step,data in enumerate(trainloader):
            
            global_step+=1
            model.train()
            optimizer.zero_grad()
            img=data['img'].cuda()                  #full size
            pc=data['pc'].cuda()
            intensity=data['intensity'].cuda()
            sn=data['sn'].cuda()
            K=data['K'].cuda()
            P=data['P'].cuda()
            pc_mask=data['pc_mask'].cuda()      
            img_mask=data['img_mask'].cuda()        #1/4 size
            B=img_mask.size(0)
            pc_kpt_idx=data['pc_kpt_idx'].cuda()    #(B,512)
            pc_outline_idx=data['pc_outline_idx'].cuda()
            img_kpt_idx=data['img_kpt_idx'].cuda()
            img_outline_idx=data['img_outline_index'].cuda()
            node_a=data['node_a'].cuda()
            node_b=data['node_b'].cuda()
            img_x=torch.linspace(0,img_mask.size(-1)-1,img_mask.size(-1)).view(1,-1).expand(img_mask.size(-2),img_mask.size(-1)).unsqueeze(0).expand(img_mask.size(0),img_mask.size(-2),img_mask.size(-1)).unsqueeze(1).cuda()
            img_y=torch.linspace(0,img_mask.size(-2)-1,img_mask.size(-2)).view(-1,1).expand(img_mask.size(-2),img_mask.size(-1)).unsqueeze(0).expand(img_mask.size(0),img_mask.size(-2),img_mask.size(-1)).unsqueeze(1).cuda()
            img_xy=torch.cat((img_x,img_y),dim=1)
            
            '''
            点云融合到图像的特征:img_features (B,64,H,W)
            点云融合到图像的相关系数:img_score (B,1,H,W)
            图像融合到点云的特征:pc_features  (B,64,H,W)
            图像融合到点云的相关系数:pc_score  (B,1,H,W)
            '''
            # img_siam_feature, pc_siam_feature, \
            # img_features, pc_features, \
            # img_feature_norm, pc_feature_norm, \
            # img_score,pc_score = model(pc,intensity,sn,img,node_a,node_b)    #64 channels feature
           
            img_siam_feature_norm, pc_siam_feature_norm, \
            img_score,pc_score = model(pc,intensity,sn,img,node_a,node_b)    #64 channels feature
            

            '''
            选出图像点云的共视区域点云坐标:pc_xyz_inline
            图像点云的共视区域点云i2p特征:pc_features_inline
            图像点云的共视区域点云i2p系数:pc_score_inline
            图像点云的不共视区域点云i2p特征:pc_features_outline
            图像点云的不共视区域点云i2p系数:pc_score_outline
            '''
           
            pc_features_inline=torch.gather(pc_siam_feature_norm,index=pc_kpt_idx.unsqueeze(1).expand(B,pc_siam_feature_norm.size(1),args.num_kpt),dim=-1)    #(B,C,num_kpt)
            pc_features_outline=torch.gather(pc_siam_feature_norm,index=pc_outline_idx.unsqueeze(1).expand(B,pc_siam_feature_norm.size(1),args.num_kpt),dim=-1)
            pc_xyz_inline=torch.gather(pc,index=pc_kpt_idx.unsqueeze(1).expand(B,3,args.num_kpt),dim=-1)
            pc_score_inline=torch.gather(pc_score,index=pc_kpt_idx.unsqueeze(1),dim=-1)         #(B,1,num_in)
            pc_score_outline=torch.gather(pc_score,index=pc_outline_idx.unsqueeze(1),dim=-1)    #(B,1,num_out)

            '''
            选出图像点云的共视区域点图像坐标:img_xy_flatten_inline
            图像点云的共视区域点图像p2i特征:pc_features_inline
            图像点云的共视区域点图像p2i系数:img_score_flatten_inline
            图像点云的不共视区域点云p2i特征:img_features_flatten_outline
            图像点云的不共视区域点云p2i系数:img_score_flatten_outline
            '''              
            # img_features_flatten=img_features.contiguous().view(img_features.size(0),img_features.size(1),-1)   #(B,C,(H*W))

            img_features_flatten = img_siam_feature_norm
            img_score_flatten=img_score.contiguous().view(img_score.size(0),img_score.size(1),-1)               #(B,1,(H*W))
            img_xy_flatten=img_xy.contiguous().view(img_siam_feature_norm.size(0),2,-1)
            img_features_flatten_inline=torch.gather(img_features_flatten,index=img_kpt_idx.unsqueeze(1).expand(B,img_features_flatten.size(1),args.num_kpt),dim=-1)
            img_xy_flatten_inline=torch.gather(img_xy_flatten,index=img_kpt_idx.unsqueeze(1).expand(B,2,args.num_kpt),dim=-1)
            img_score_flatten_inline=torch.gather(img_score_flatten,index=img_kpt_idx.unsqueeze(1),dim=-1)
            img_features_flatten_outline=torch.gather(img_features_flatten,index=img_outline_idx.unsqueeze(1).expand(B,img_features_flatten.size(1),args.num_kpt),dim=-1)
            img_score_flatten_outline=torch.gather(img_score_flatten,index=img_outline_idx.unsqueeze(1),dim=-1)
            

            #用真实位姿，计算共视区域点云的二维图像投影
            pc_xyz_projection=torch.bmm(K,(torch.bmm(P[:,0:3,0:3],pc_xyz_inline)+P[:,0:3,3:]))
            #pc_xy_projection=torch.floor(pc_xyz_projection[:,0:2,:]/pc_xyz_projection[:,2:,:]).float()
            pc_xy_projection=pc_xyz_projection[:,0:2,:]/pc_xyz_projection[:,2:,:]

            '''
            这有一点类似coarse点对匹配的感觉,计算共视的图像点与点云点,计算两两的重投影误差
            根据重投影误差是否小于args.dist_thres确定点云-图像相关mask(B,N,N),这步mask的计算使用的全部为真值
            '''
            correspondence_mask=(torch.sqrt(torch.sum(torch.square(img_xy_flatten_inline.unsqueeze(-1)-pc_xy_projection.unsqueeze(-2)),dim=1))<=args.dist_thres).float()
            
            #计算负样本loss

            img_xy_flatten_outline = torch.gather(img_xy_flatten,index=img_outline_idx.unsqueeze(1).expand(B,2,args.num_kpt),dim=-1)
            pc_xyz_outline=torch.gather(pc,index=pc_outline_idx.unsqueeze(1).expand(B,3,args.num_kpt),dim=-1)
            pc_xyz_projection_outline=torch.bmm(K,(torch.bmm(P[:,0:3,0:3],pc_xyz_outline)+P[:,0:3,3:]))
            pc_xy_projection_outline=pc_xyz_projection_outline[:,0:2,:]/pc_xyz_projection_outline[:,2:,:]
            neg_correspondence_mask = (torch.sqrt(torch.sum(torch.square(img_xy_flatten_outline.unsqueeze(-1)-pc_xy_projection_outline.unsqueeze(-2)),dim=1))<=args.dist_thres).float()
            neg_siam_loss = siam_loss(img_features_flatten_outline, pc_features_outline, neg_correspondence_mask, 1.0)
            # neg_siam_loss = neg_siam_loss2(img_features_flatten_outline, pc_features_outline, neg_correspondence_mask, 2.0)

            
            #根据features计算loss
            # loss_desc,dists=desc_loss(img_features_flatten_inline,pc_features_inline,correspondence_mask,pos_margin=args.pos_margin,neg_margin=args.neg_margin)
            siamese_loss = siam_loss(img_features_flatten_inline, pc_features_inline, correspondence_mask, 1.0)
            # siamese_loss = siam_loss2(img_features_flatten_inline, pc_features_inline, correspondence_mask, 2.0)

            #loss_det=loss2.det_loss(img_score_flatten_inline.squeeze(),img_score_flatten_outline.squeeze(),pc_score_inline,pc_score_outline.squeeze())
            #根据相关系数计算loss2
            #就是让inline的(1-score)+outlin的score，当inline_score越大，outline_score越小的时候loss越小
            loss_det=det_loss2(img_score_flatten_inline.squeeze(),img_score_flatten_outline.squeeze(),pc_score_inline.squeeze(),pc_score_outline.squeeze())
            loss =  neg_siam_loss * 0  + siamese_loss  + loss_det 
            #loss=loss_desc
            
            loss.backward()
            optimizer.step()
            
            #torch.cuda.empty_cache()

            if global_step%16==0:
                logger.info('%s-%d-%d, loss: %f, siamese_loss: %f, loss det: %f, neg_siamese_loss: %f'%('train',epoch,global_step,loss.data.cpu().numpy(),siamese_loss.data.cpu().numpy(),loss_det.data.cpu().numpy(), neg_siam_loss.data.cpu().numpy()))
                # logger.info('%s-%d-%d, loss: %f, siamese_loss: %f, loss det: %f'%('train',epoch,global_step,loss.data.cpu().numpy(),siamese_loss.data.cpu().numpy(),loss_det.data.cpu().numpy()))

            if global_step%args.val_freq==0 and epoch>5:
                t_diff, r_diff, t_val_diff, r_val_diff, success_rate=test_acc(model,testloader,args)
                if t_diff<=best_t_diff:
                    torch.save(model.state_dict(),os.path.join(logdir,'mode_best_t.t7'))
                    best_t_diff=t_diff
                if r_diff<=best_r_diff:
                    torch.save(model.state_dict(),os.path.join(logdir,'mode_best_r.t7'))
                    best_r_diff=r_diff
                logger.info('%s-%d-%d, t_error: %f, r_error: %f, t_val_diff: %f, r_val_diff:%f, success_rate: %f'%('test',epoch,global_step,t_diff,r_diff, t_val_diff, r_val_diff, success_rate))
                torch.save(model.state_dict(),os.path.join(logdir,'mode_last.t7'))
        
        if epoch%5==0 and epoch>5:
            current_lr=current_lr*0.25
            if current_lr<args.min_lr:
                current_lr=args.min_lr
            for param_group in optimizer.param_groups:
                param_group['lr']=current_lr
            logger.info('%s-%d-%d, updata lr, current lr is %f'%('train',epoch,global_step,current_lr))
        torch.save(model.state_dict(),os.path.join(logdir,'mode_epoch_%d.t7'%epoch))