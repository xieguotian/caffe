import caffe
import numpy as np
from caffe.proto.caffe_pb2 import LayerParameter,NetParameter
from google.protobuf import text_format
from collections import OrderedDict
import sys

# conv FLOPs, only consider convolution with w, no bias
def conv_FLOPs(param,input_size):
    kernel_size = param[0]
    stride = param[1]
    pad = param[2]
    num_output= param[3]
    group = param[4]

    [in_n,in_ch,in_h,in_w] = input_size

    out_n = in_n
    out_ch = num_output
    out_h = (in_h + 2*pad - kernel_size)/stride + 1
    out_w = (in_w + 2*pad - kernel_size)/stride + 1

    FLOPs = out_ch*(in_ch*kernel_size*kernel_size)*out_h*out_w
    return FLOPs

def inner_FLOPs(param,input_size):
    in_n = input_size[0]
    in_ch = input_size[1]
    num_output = param
    out_ch = num_output

    FLOPs = out_ch*in_ch
    return FLOPs
def get_conv_param(conv_param):
    pad = 0
    stride = 1
    kernel_size = 1
    num_out = 0
    group = 1
    if len(conv_param.pad)>0:
        pad = conv_param.pad[0]
    if len(conv_param.stride)>0:
        stride = conv_param.stride[0]
    if len(conv_param.kernel_size)>0:
        kernel_size = conv_param.kernel_size[0]
    #if conv_param.has_num_output():
    num_out = conv_param.num_output
    #if conv_param.has_group():
    group = conv_param.group

    return np.array([kernel_size,stride,pad,num_out,group])

if len(sys.argv)<2:
    print 'usage: python net_FLOP_compute.py net_file'
net_file = sys.argv[1]
#net_file = 'vgg_train_16_layers.prototxt'

net_param = NetParameter()
with open(net_file,'r') as fid:
    text_format.Merge(str(fid.read()),net_param)
layer_dict = OrderedDict()

for idx,layer in enumerate(net_param.layer):
    layer_dict[layer.name] = idx

net = caffe.Net(net_file,caffe.TEST)

total_FLOPs = np.double(0)

input_size = 0
for idx,name in enumerate(net._layer_names):
    print name,'\t',
    if net.layers[idx].type=='Convolution':
        input_size = net._blobs[net._bottom_ids(idx)[0]].data.shape
        conv_param = get_conv_param(net_param.layer[layer_dict[name]].convolution_param)
        tmp_FLOPs  = conv_FLOPs(conv_param,input_size)
    elif net.layers[idx].type=='InnerProduct':
        input_size = net._blobs[net._bottom_ids(idx)[0]].data.shape
        num_out = net_param.layer[layer_dict[name]].inner_product_param.num_output
        tmp_FLOPs = inner_FLOPs(num_out,input_size)
    else:
        if len(net._bottom_ids(idx))>0:
            input_size = net._blobs[net._bottom_ids(idx)[0]].data.shape
        tmp_FLOPs = 0
    total_FLOPs += tmp_FLOPs
    print tmp_FLOPs,input_size

print 'total FLOPs is: {:,}'.format(np.int64(total_FLOPs))