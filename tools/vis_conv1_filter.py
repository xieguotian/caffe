import caffe
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from cv_util.b64_visualize import show_tile
from cv_util.image_util import resize_image_preserve_ratio

def get_conv1_filter(param):
    param_images = []
    for i in range(param.shape[0]):
        img = param[i,:,:,:]
        img = np.transpose(img,(1,2,0))
        img = img - img.min()
        img = img / img.max()
        img = img*255
        img = img.astype(np.uint8)
        param_images.append(img)
        #print img

    show_filters = show_tile(param_images,100,1)[:,:,::-1]
    return show_filters

if __name__=="__main__":
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
        "--save_png",
        type=str,
        default="filter.png",
        help="save name of visualization result"
    )
    args = parser.parse_args()
    net_proto = args.net_proto
    net_param = args.net_param
    layer_name = args.layer_name

    save_name = layer_name + "_" + args.save_png

    net = caffe.Net(net_proto,net_param,caffe.TEST)
    param = net.params[layer_name][0].data
    show_filters = get_conv1_filter(param)
    plt.imsave(save_name,show_filters)