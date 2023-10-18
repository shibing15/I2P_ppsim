import numpy as np
import os
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

base_path='./siam_result14_error_1.00_1.00_pos_margin_0.20_neg_margin_1.80'

total_num=len(os.listdir(base_path))//5
print(total_num)
t_error_set=[]
r_error_set=[]
t_error_all_set=[]
r_error_all_set=[]
eval_t_error_set =[]
eval_r_error_set = []

for i in range(total_num):
    if i == 398:
        continue
    t_error_set.append(np.load(os.path.join(base_path,'t_error_%d.npy'%i)))
    r_error_set.append(np.load(os.path.join(base_path, 'angle_error_%d.npy' % i)))
    eval_t_error = np.load(os.path.join(base_path,'t_error_%d.npy'%i))
    eval_r_error = np.load(os.path.join(base_path, 'angle_error_%d.npy' % i))
    for j in range(eval_t_error.shape[0]):
        if (eval_t_error[j] < 7) & (eval_r_error[j] < 15):
            eval_t_error_set.append(eval_t_error[j])
            eval_r_error_set.append(eval_r_error[j])
        t_error_all_set.append(eval_t_error[j])
        r_error_all_set.append(eval_r_error[j])

print('eval number: %d' % len(eval_t_error_set))
print('RTE %0.4f +- %0.4f'%( np.mean(np.array(eval_t_error_set)), np.std(np.array(eval_t_error_set))))
print('RRE %0.4f +- %0.4f'%(np.mean(np.array(eval_r_error_set)),np.std(np.array(eval_r_error_set))))
print('total number: %d' %len(t_error_all_set))
# import ipdb; ipdb.set_trace()
print('RTE %0.4f +- %0.4f'%(np.mean(np.array(t_error_all_set)), np.std(np.array(t_error_all_set))))
print('RRE %0.4f +- %0.4f'%(np.mean(np.array(r_error_all_set)), np.std(np.array(r_error_all_set))))            
