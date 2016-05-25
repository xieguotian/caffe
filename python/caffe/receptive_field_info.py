import numpy as np
import cv2

def receptive_field_info(param):
    info = np.zeros(param.shape[0],3)
    info[0,0] = param[0,0]
    info[0,1] = param[0,1]
    info[0,2] = param[0,2]

    for i in range(1,info.shape[0]):
        info[i,0] = (param[i,0]-1)*info[i-1,1]+info[i-1,0]
        info[i,1] = param[i,1]*info[i-1,1]
        info[i,2] = param[i-1,2]-param[i,2]*info[i-1,1]
    return info


def filter_img_vis(net_param,layer_idx,samp_idx,img_list,img_root):
    net_info = receptive_field_info(net_param)
    kernel = net_info[layer_idx,0]
    stride = net_info[layer_idx,1]
    pad = net_info[layer_idx,2]

    top_num = samp_idx.shape[1] / 3.0
    img_size = 100

    top_imgs = np.zeros((img_size,img_size,3,top_num,samp_idx.shape[0]),np.uint8)

    for i in range(samp_idx.shape[0]):
        for j in range(0,samp_idx.shape[1],3):
            img_idx = samp_idx[i,j]
            im_name = img_root + '/' + img_list[img_idx]
            img = cv2.imread(im_name)
            img_size = img.shape

            w_idx = samp_idx[i,j+2]
            h_idx = samp_idx[i,j+1]

            h_st = np.max((h_idx-1)*stride+pad,0)+1
            w_st = np.max((w_idx-1)*stride+pad,0)+1
            h_end = min(h_st+kernel-1,img_size[0])
            w_end = min(w_st+kernel-1,img_size[1])
            img = img[h_st:h_end,w_st:w_end,:]

