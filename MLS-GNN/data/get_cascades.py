
import torch
import numpy as np
import os



def load_cascades(cascade_dir, device,trans=False, final_t1=False,final_t2=False,final_t3 = False):
    cas = []
    if final_t1:
        cas.append(np.genfromtxt(cascade_dir.parent.joinpath('output_t1.txt')))#np.genfromtxt读取文件
    elif final_t2:
        cas.append(np.genfromtxt(cascade_dir.parent.joinpath('output_t2.txt')))#np.genfromtxt读取文件
    elif final_t3:
        cas.append(np.genfromtxt(cascade_dir.parent.joinpath('output_t3.txt')))
    else:
        cas_list = os.listdir(cascade_dir)
        cas_list.sort(key=lambda x: int(x[:-4]))
        cas.append(np.genfromtxt(cascade_dir.joinpath(cas_list[-1])))
    cas = torch.FloatTensor(cas)
    if trans:
        cas = torch.transpose(cas, 1, 2)
    cas = cas.to(device)
    return cas


def remove_overfitting_cascades(cascade_dir, patience):
    cas_list = os.listdir(cascade_dir)
    cas_list.sort(key=lambda x: int(x[:-4]))
    for i in range(patience):
        os.remove(cascade_dir.joinpath(cas_list[-1-i]))
