import caffe
import numpy as np
import argparse
import threading
from multiprocessing import Queue
import matplotlib.pyplot as plt
import cv2
from cv_util.b64_visualize import show_tile
from cv_util.image_util import resize_image_preserve_ratio
import pickle

def sort_neuron_top9(info_queue):
    """thread for sort the neuron of top9"""
    top9_info = [[],[],[],[]]
    while True:
        neuron_info_new = info_queue.get()
        if len(neuron_info_new)==0:
            break

        all_info = [[],[],[],[]]
        for i in range(3):
            all_info[i].extend(top9_info[i])
            all_info[i].extend(neuron_info_new[i])
        all_info[3].extend(top9_info[3])
        all_info[3].extend(neuron_info_new[3])

        for i in range(3):
            all_info[i] = np.array(all_info[i])

        neuron_idx = np.argsort(all_info[0],axis=0)[::-1,:][:9,:]

        top9_info = [[],[],[],[]]
        top9_info[3].extend(all_info[3])

        for tidx in range(neuron_idx.shape[0]):
            sel_idx = neuron_idx[tidx,:]*neuron_idx.shape[1]+np.arange(neuron_idx.shape[1])
            top9_info[0].append(all_info[0].flat[sel_idx])
            tmp_shape = all_info[1].shape
            top9_info[1].append(all_info[1].reshape((tmp_shape[0]*tmp_shape[1],tmp_shape[2]))[sel_idx,:])
            top9_info[2].append(all_info[2].flat[sel_idx])
    info_queue.put(top9_info)

parser = argparse.ArgumentParser()
parser.add_argument(
    "net_proto",
    help="net proto structure file"
)
parser.add_argument(
    "net_param",
    help="net weights param file"
)
parser.add_argument(
    "layer_name",
    help="layer_name for visualization"
)
parser.add_argument(
    "--dataset",
    type=str,
    default="D:\\users\\v-guoxie\\v-guoxie\\data\\ILSVRC2012\\ILSVRC2012\\val\\",
    help="dataset folder"
)
parser.add_argument(
    "--img_list",
    type=str,
    default="\\\\msra-sms40/v_guoxie/train_org_val_key.txt",
    help="data list"
)
parser.add_argument(
    "--save_png",
    type=str,
    default="vis.png",
    help="save name of visualization result"
)
parser.add_argument(
    "--is_vis_diff",
    action='store_true',
    help="visualizae diff or not"
)
parser.add_argument(
    "--blob_name",
    type=str,
    default="",
    help="name of top blob"
)
parser.add_argument(
    "--device_id",
    type=int,
    default=0,
    help="device id"
)

args = parser.parse_args()
net_proto = args.net_proto
net_param = args.net_param
layer_name = args.layer_name
dataset = args.dataset
img_list = args.img_list
save_name = layer_name + "_" + args.save_png
is_vis_diff = args.is_vis_diff
blob_name = args.blob_name
if blob_name=="":
    blob_name=layer_name
device_id = args.device_id

# load net and read receptive field info
caffe.set_mode_gpu()
caffe.set_device(device_id)
net = caffe.Net(net_proto,net_param,caffe.TEST)
info = net.get_field_size(layer_name)
kernel_size = info[0]
stride = info[1]
pad = info[2]

neuron_value_all = []
side_info_all = []
max_idx_all = []
sel_img_idx_all = []

neuron_queue = Queue(5)
neuron_thread = threading.Thread(target=sort_neuron_top9,args=(neuron_queue,))
neuron_thread.start()

with open(img_list) as fid:
    for ix,line in enumerate(fid):
        #if ix>2000:
            #break
        img_name = line.split('\t')[0].strip()
        image_name = dataset + img_name[9:]
        net._set_input_image(image_name,0)
        net.forward()
        neuron_val = net.blobs[blob_name].data[0].copy()

        out_shape = neuron_val.shape
        print out_shape
        neuron_val = neuron_val.reshape((out_shape[0],out_shape[1]*out_shape[2]))
        print (neuron_val[:,0]*np.log(neuron_val[:,0]+0.000000000001)).sum()
        print neuron_val.sum()
        plt.plot(neuron_val)
        plt.show()
        max_idx = neuron_val.argmax(axis=1)
        tmp_max_idx = max_idx + np.arange(out_shape[0])*(out_shape[1]*out_shape[2])
        max_idx = np.array(np.unravel_index(tmp_max_idx,out_shape)).transpose()
        max_neuron_val = neuron_val.flat[tmp_max_idx]

        neuron_value_all.append(max_neuron_val)
        max_idx_all.append(max_idx)
        sel_img_idx_all.append(np.zeros((max_idx.shape[0],))+ix)
        side_info_all.append([out_shape,img_name])
        #side_info_all.append([np.array(max_idx),out_shape,img_key])

        if (ix+1)%100==0:
            print "process %d"%(ix+1)
            # push info into queue.
            neuron_queue.put([neuron_value_all,max_idx_all,sel_img_idx_all,side_info_all])
            neuron_value_all = []
            max_idx_all = []
            sel_img_idx_all = []
            side_info_all = []

    neuron_queue.put([])
    neuron_thread.join()

    top9_info = neuron_queue.get()
    top9_neuron_value = np.array(top9_info[0])
    top9_max_idx = np.array(top9_info[1])
    top9_sel_img_idx = np.array(top9_info[2])
    top9_side_info = np.array(top9_info[3])

    all_top9_patch = []
    all_top9_diff  = []
    for ch in range(top9_neuron_value.shape[1]):
        top9_patch = []
        top9_diff = []
        for i in range(top9_neuron_value.shape[0]):
            max_idx = top9_max_idx[i,ch,:]
            neuron_value = top9_neuron_value[i,ch]
            out_shape,img_name = top9_side_info[top9_sel_img_idx[i,ch]]
            image_name = dataset + img_name[9:]

            img = cv2.imread(image_name,cv2.CV_LOAD_IMAGE_COLOR)
            img = resize_image_preserve_ratio(img,256)

            h_idx = max_idx[1]
            w_idx = max_idx[2]
            h_st = min(max((h_idx)*stride+pad,0),img.shape[0]-1)
            w_st = min(max((w_idx)*stride+pad,0),img.shape[1]-1)
            h_end = min((h_idx)*stride+pad+kernel_size,img.shape[0])
            w_end = min((w_idx)*stride+pad+kernel_size,img.shape[1])

            img = img[h_st:h_end,w_st:w_end,:]

            if not img.shape[0]*img.shape[1]*img.shape[2]==0:
                top9_patch.append(img)
            else:
                top9_patch.append(np.zeros((100,100,3)))

            if is_vis_diff:
                net._set_input_image(image_name,0)
                net.forward()
                net.blobs[blob_name].diff[...] = 0
                net.blobs[blob_name].diff[0,ch,h_idx,w_idx] = neuron_value
                #if (not sel_val>0):
                    #print sel_idx
                #net.blobs[layer_name].diff[...] = net.blobs[layer_name].data
                net.backward(start=layer_name)
                input_diff = net.blobs['data'].diff.copy()
                #input_diff = np.sqrt((input_diff[0]**2).sum(axis=0))

                input_diff = np.transpose(input_diff[0],[1,2,0])
                input_diff = input_diff - input_diff.min()
                input_diff = input_diff / input_diff.max()
                input_diff = input_diff ** 2
                input_diff = input_diff / input_diff.max()

                input_diff = (input_diff[h_st:h_end,w_st:w_end,:]*255).astype(np.uint8)

                if not input_diff.shape[0]*input_diff.shape[1]==0:
                    top9_diff.append(input_diff)
                else:
                    top9_diff.append(np.zeros((100,100,3)))

        tmp_img = show_tile(top9_patch,100,1)[:,:,::-1]
        all_top9_patch.append(tmp_img)
        if is_vis_diff:
            tmp_diff = show_tile((top9_diff),100,1)[:,:,::-1]
            all_top9_diff.append(tmp_diff)

    tmp_img = show_tile(all_top9_patch,all_top9_patch[0].shape[0],5)
    plt.imsave(save_name,tmp_img)
    if is_vis_diff:
        diff_img = show_tile(all_top9_diff,all_top9_diff[0].shape[0],5)
        plt.imsave(save_name.replace(".png","_diff.png"),diff_img)
    with open(save_name.replace(".png","_top9.pickle"),'wb') as fid:
        pickle.dump(top9_info,fid)
