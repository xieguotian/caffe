import sys
import caffe
import numpy as np
from caffe.proto.caffe_pb2 import LayerParameter,NetParameter
from google.protobuf import text_format
from collections import OrderedDict

# input_size: [n,ch,h,w]
# param: [kernel,stride,pad,num,group]
def conv_mem(param,input_size):
    in_n = input_size[0]
    in_ch = input_size[1]
    in_h = input_size[2]
    in_w = input_size[3]

    kernel_size = param[0]
    stride = param[1]
    pad = param[2]
    num_out = param[3]
    group = param[4]

    out_n = in_n
    out_ch = num_out
    out_h = (in_h + 2*pad - kernel_size)/stride + 1
    out_w = (in_w + 2*pad - kernel_size)/stride + 1

    top_size = np.array([out_n,out_ch,out_h,out_w]).astype(np.int32)
    top_mem = out_n*out_ch*out_h*out_w * 2 # data and diff
    cache_mem = (in_ch/group)*kernel_size*kernel_size*out_h*out_w  # data
    param_mem = in_ch*kernel_size*kernel_size*out_ch * 2
    return [top_size,top_mem,cache_mem,param_mem]

# input_size: [n,ch,h,w]
# param: [kernel,stride,pad]
def pooling_mem(param,input_size):
    in_n = input_size[0]
    in_ch = input_size[1]
    in_h = input_size[2]
    in_w = input_size[3]

    kernel_size = param[0]
    stride = param[1]
    pad = param[2]

    out_n = in_n
    out_ch = in_ch
    out_h = (in_h + 2*pad - kernel_size)/stride + 1
    out_w = (in_w + 2*pad - kernel_size)/stride + 1

    top_size = np.array([out_n,out_ch,out_h,out_w]).astype(np.int32)
    top_mem = out_n*out_ch*out_h*out_w * 2 # data and diff
    cache_mem = out_n*out_ch*out_h*out_w # data
    param_mem = 0
    return [top_size,top_mem,cache_mem,param_mem]

def scale_mem(input_size):
    top_size = np.array(input_size).astype(np.int32)
    top_mem = 0
    cache_mem = top_size[0]*top_size[1]*top_size[2]*top_size[3] # data
    param_mem = 0
    return [top_size,top_mem,cache_mem,param_mem]

def bn_mem(input_size):
    top_size = np.array(input_size).astype(np.int32)
    top_mem = 0
    # two cache data and one diff
    cache_mem = top_size[0]*top_size[1]*top_size[2]*top_size[3] + 2*top_size[0]*top_size[1]*top_size[2]*top_size[3]
    param_mem = 0
    return [top_size,top_mem,cache_mem,param_mem]

def inner_mem(param,input_size):
    top_size = np.array([input_size[0],param,1,1]).astype(np.int32)
    top_mem = top_size[0]*top_size[1] * 2 # data and iff
    cache_mem = 0
    param_mem = top_size[1]*input_size[1] *2
    return [top_size,top_mem,cache_mem,param_mem]

def lrn_mem(input_size):
    top_size = np.array(input_size).astype(np.int32)
    top_mem = top_size[0]*top_size[1]*top_size[2]*top_size[3]*2
    cache_mem = 0
    param_mem = 0
    return [top_size,top_mem,cache_mem,param_mem]

def lrn_mem(input_size):
    top_size = np.array(input_size).astype(np.int32)
    top_mem = top_size[0]*top_size[1]*top_size[2]*top_size[3]*2
    cache_mem = 0
    param_mem = 0
    return [top_size,top_mem,cache_mem,param_mem]

def split_mem(input_size):
    top_size = np.array(input_size).astype(np.int32)
    top_mem=2
    for size_v in top_size:
        top_mem *= size_v
    #top_mem = top_size[0]*top_size[1]*top_size[2]*top_size[3]*2 # two diff
    cache_mem = 0
    param_mem = 0
    return [top_size,top_mem,cache_mem,param_mem]

def eltwise_mem(param,input_size):
    if param=='MAX':
        top_mem = 3 #data and diff and a mask idx
    else:
        top_mem = 2 #data and diff
    top_size = input_size
    for v in top_size:
        top_mem *= v
    cache_mem = 0
    param_mem = 0
    return [top_size,top_mem,cache_mem,param_mem]

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

def get_pool_param(pool_param):
    pad = 0
    stride = pool_param.stride
    kernel_size = pool_param.kernel_size
    return np.array([kernel_size,stride,pad])

'''
# kernel_size,stride,pad,num_out,group,layer_type
# layer_type:
#   0: conv
#   1: pooling
#   2: scale
#   3: bn
#   4: inner
#   5: lrn
net_param = np.array([[11,4,0,96,1,0],
                      [0,0,0,0,1,5],
                      [3,2,0,0,0,1],
                      [5,1,2,256,2,0],
                      [0,0,0,0,1,5],
                      [3,2,0,0,0,1],
                      [3,1,1,384,1,0],
                      [3,1,1,384,2,0],
                      [3,1,1,256,1,0],
                      [3,2,0,0,1,1],
                      [0,0,0,4096,1,4],
                      [0,0,0,4096,1,4],
                      [0,0,0,1000,1,4]]).astype(np.int)

net_param = np.array([[3,1,1,64,1,0],
                      [3,1,1,64,1,0],
                      [2,2,0,0,1,1],
                      [3,1,1,128,1,0],
                      [3,1,1,128,1,0],
                      [2,2,0,0,1,1],
                      [3,1,1,256,1,0],
                      [3,1,1,256,1,0],
                      [3,1,1,256,1,0],
                      [2,2,0,0,1,1],
                      [3,1,1,512,1,0],
                      [3,1,1,512,1,0],
                      [3,1,1,512,1,0],
                      [2,2,0,0,1,1],
                      [3,1,1,512,1,0],
                      [3,1,1,512,1,0],
                      [3,1,1,512,1,0],
                      [2,2,0,0,1,1],
                      [0,0,0,4096,1,4],
                      [0,0,0,4096,1,4],
                      [0,0,0,1000,1,4]
])
'''
net_name = 'res_152'
net_file = '\\\\GCR\\\scratch\\B99\\v-guoxie\\proto\\residual_net\\res_net_152.prototxt'
with open('record.txt','a+') as fid_out:
    net_param = NetParameter()
    with open(net_file,'r') as fid:
        text_format.Merge(str(fid.read()),net_param)
    layer_dict = OrderedDict()

    for idx,layer in enumerate(net_param.layer):
        layer_dict[layer.name] = idx

    net = caffe.Net(net_file,caffe.TEST)


    total_mem = np.double(0)
    total_top_vec =np.double(0)
    total_param_mem = np.double(0)
    # input_data memory
    input_size = np.array(net.blobs['data'].data.shape)
    input_size[0] =256
    print input_size
    total_mem += input_size[0]*input_size[1]*input_size[2]*input_size[3]
    total_top_vec += total_mem

    share_dict = OrderedDict()
    share_name = []
    share_size = []
    for idx,name in enumerate(net._layer_names):
        print name,'\t',
        if net.layers[idx].type=='Convolution':
            input_size = np.array(net._blobs[net._bottom_ids(idx)[0]].data.shape)
            input_size[0] = 256
            conv_param = get_conv_param(net_param.layer[layer_dict[name]].convolution_param)
            print conv_param,'\t',
            [top_size,top_mem,cache_mem,param_mem] = conv_mem(conv_param,input_size)

            if net_param.layer[layer_dict[name]].mem_opt_type!='none':
                opt_type = net_param.layer[layer_dict[name]].mem_opt_type
                if share_dict.has_key(opt_type):
                    top_mem = top_mem/2.0
                    share_name.append(name)
                    share_size.append(top_size)
                else:
                    share_dict[opt_type] = 1

        elif net.layers[idx].type=='Pooling':
            input_size = np.array(net._blobs[net._bottom_ids(idx)[0]].data.shape)
            input_size[0] = 256
            pool_param = get_pool_param(net_param.layer[layer_dict[name]].pooling_param)
            print pool_param,'\t',
            [top_size,top_mem,cache_mem,param_mem] = pooling_mem(pool_param,input_size)
        elif net.layers[idx].type=='InnerProduct':
            input_size = np.array(net._blobs[net._bottom_ids(idx)[0]].data.shape)
            input_size[0] = 256
            num_out = net_param.layer[layer_dict[name]].inner_product_param.num_output
            [top_size,top_mem,cache_mem,param_mem] = inner_mem(num_out,input_size)
        elif net.layers[idx].type=='BatchNorm':
            input_size = np.array(net._blobs[net._bottom_ids(idx)[0]].data.shape)
            input_size[0] = 256
            [top_size,top_mem,cache_mem,param_mem] = bn_mem(input_size)
            total_param_mem+=cache_mem
        elif net.layers[idx].type=='Scale':
            input_size = np.array(net._blobs[net._bottom_ids(idx)[0]].data.shape)
            input_size[0] = 256
            [top_size,top_mem,cache_mem,param_mem] = scale_mem(input_size)
        elif net.layers[idx].type=='Split':
            input_size = np.array(net._blobs[net._bottom_ids(idx)[0]].data.shape)
            input_size[0] = 256
            [top_size,top_mem,cache_mem,param_mem] = split_mem(input_size)
        elif net.layers[idx].type=='LRN':
            input_size = np.array(net._blobs[net._bottom_ids(idx)[0]].data.shape)
            input_size[0] = 256
            [top_size,top_mem,cache_mem,param_mem] = lrn_mem(input_size)
        elif net.layers[idx].type=='Eltwise':
            input_size = np.array(net._blobs[net._bottom_ids(idx)[0]].data.shape)
            input_size[0] = 256
            param_elt = net_param.layer[layer_dict[name]].eltwise_param.operation
            [top_size,top_mem,cache_mem,param_mem] = eltwise_mem(param_elt,input_size)
            top_mem = 0
        else:
            print 0
            continue
        print input_size,'\t',
        print top_mem,'\t',
        print cache_mem,'\t',
        print param_mem
        total_mem += top_mem
        total_mem += cache_mem
        total_mem += param_mem
        total_top_vec += top_mem

    print 'total memory %f GB' % (total_mem*4/1024.0/1024.0/1024.0)
    print 'top_vec memory %f GB' % (total_top_vec*4/1024.0/1024.0/1024.0)
    print 'param %f GB' %(total_param_mem*4/1024.0/1024.0/1024.0)
    print >>fid_out,'{} total memory: {}'.format(net_name,total_mem*4/1024.0/1024.0/1024.0)
    for name,sz in zip(share_name,share_size):
        print >>fid_out,name,sz
'''
for row in range(net_param.shape[0]):
    if net_param[row,5]==0:
        print 0
        [top_size,top_mem,cache_mem,param_mem] = conv_mem(net_param[row,:5],input_size)
    elif net_param[row,5]==1:
        print 1
        [top_size,top_mem,cache_mem,param_mem] = pooling_mem(net_param[row,:3],input_size)
    elif net_param[row,5]==2:
        print 2
        [top_size,top_mem,cache_mem,param_mem] = scale_mem(input_size)
    elif net_param[row,5]==3:
        print 3
        [top_size,top_mem,cache_mem,param_mem] = bn_mem(input_size)
    elif net_param[row,5]==4:
        print 4
        input_size = np.array([input_size[0],input_size[1]*input_size[2]*input_size[3],1,1]).astype(np.int32)
        [top_size,top_mem,cache_mem,param_mem] = inner_mem(net_param[row,3],input_size)
    elif net_param[row,5]==5:
        print 5
        [top_size,top_mem,cache_mem,param_mem] = lrn_mem(input_size)
    input_size = top_size
    print input_size
    print top_mem
    print cache_mem
    print param_mem
    total_mem += top_mem
    total_mem += cache_mem
    total_mem += param_mem
    total_top_vec += top_mem

print 'total memory %f GB' % (total_mem*4/1024.0/1024.0/1024.0)
print 'top_vec memory %f GB' % (total_top_vec*4/1024.0/1024.0/1024.0)

'''