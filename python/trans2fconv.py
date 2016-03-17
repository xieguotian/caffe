import numpy as np
import caffe
import sys
import os
if not len(sys.argv)==5:
    print 'Usage: python trans2fconv.py net_origin net_fconv net_param layers_set'

caffe.set_device(0)
net_org = sys.argv[1]
net_fconv = sys.argv[2]
net_param = sys.argv[3]
layers_set = sys.argv[4]

layers = layers_set.split(',')

net2 = caffe.Net(net_fconv, net_param, caffe.TEST)
net1 = caffe.Net(net_org, net_param, caffe.TEST)

for layer in layers:
    print 'layer: '+ layer
    net2.params[layer+'_conv'][0].data.flat = net1.params[layer][0].data.flatten().copy()
    net2.params[layer+'_conv'][1].data.flat = net1.params[layer][1].data.flatten().copy()

net2.save('fconv_'+os.path.basename(net_param))
