import numpy as np
import caffe
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "net_proto",
    help="net prototxt file"
)
parser.add_argument(
    "net_proto_mask",
    help="net prototxt file"
)
parser.add_argument(
    "net_param",
    help="net param file"
)
parser.add_argument(
    "layer_name",
    help="layer name to transfrom from conv to FIF"
)
parser.add_argument(
    "--net_param_mask",
    type=str,
    default="",
    help="net param file for mask"
)

args = parser.parse_args()

layer_name = args.layer_name
net_proto = args.net_proto
net_param = args.net_param
net_proto_mask = args.net_proto_mask
net_param_mask = net_param_mask
if net_param_mask=="":
    net_param_mask = net_param.replace(".caffemodel","_mask.caffemodel")

caffe.set_mode_gpu()
net1 = caffe.Net(net_proto,net_param,caffe.TEST)
param_w = net1.params[name][0].data
param_b = net1.params[name][1].data
del net1

net2 = caffe.Net(net_proto_mask,net_param,caffe.TEST)

param_w = np.transpose(param_w,[2,3,0,1])
h,w,n_out,n_in = param_w.shape
param_w = param_w.reshape((h*w*n_out,n_in,1,1))
net2.params[name2][0].data[...] = param_w

param_b = np.tile(param_b[:,np.newaxis],(1,9)).flatten() / 9.0
net2.params[name2][1].data[...] = param_b

net2.save(net_param_mask)