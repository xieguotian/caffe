import caffe
import sys
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "net_proto",
    help="prototxt of net work"
)
parser.add_argument(
    "save_prefix",
    help="prefix of save file name."
)
parser.add_argument(
    "--net_param",
    default="",
    help="net param"
)
parser.add_argument(
    "--not_set_zeros_lr_mult",
    action='store_true',
    help="set zeros of lr mult"
)

args = parser.parse_args()

net_proto = args.net_proto
net_param = args.net_param
pos_fix = args.save_prefix
set_zeros = not args.not_set_zeros_lr_mult

base_name = os.path.basename(net_proto).strip().split('.')
net_new_proto = base_name[0]+'_'+pos_fix+'.'+base_name[1]

with open(net_proto) as fid:
    with open(net_new_proto,'w') as fout:
        for line in fid:
            line = line.strip('\n')
            if set_zeros:
                if 'lr_mult' in line or 'decay_mult' in line:
                    pos = line.rfind(':')+1
                    line = line[:pos] + '0.0'
                if 'BatchNormTorch' in  line:
                    print >>fout, '\tbatch_norm_param{ use_global_stats: true }'
            if 'bottom' in line or 'top' in line or 'name' in line:
                if not 'data' in line:
                    pos = line.rfind('\"')
                    line = line[:pos] + '_' + pos_fix +line[pos:]
            print >>fout,line

if net_param!="":
    net = caffe.Net(net_proto,net_param,caffe.TEST)
    all_params = net.params.items()
    del net
    net2 =  caffe.Net(net_new_proto,caffe.TEST)

    for name,param in all_params:
        print 'copy param '+name
        new_name =name+'_' + pos_fix
        for ix,sub_param  in enumerate(param):
            net2.params[new_name][ix].data[:] = sub_param.data.copy()

    base_name = os.path.basename(net_param).strip().split('.')
    net_new_param = base_name[0]+'_'+pos_fix+'.'+base_name[1]
    net2.save(net_new_param)
