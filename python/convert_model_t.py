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
    "--sub",
    action='store_true',
    help="redraw"
)

args = parser.parse_args()

net_proto = args.net_proto
net_param = args.net_param
pos_fix = args.save_prefix
is_sub = args.sub

base_name = os.path.basename(net_proto).strip().split('.')
net_new_proto = base_name[0]+'_'+pos_fix+'.'+base_name[1]

with open(net_proto) as fid:
    with open(net_new_proto,'w') as fout:
        for line in fid:
            line = line.strip('\n')
            if 'bottom' in line or 'top' in line or 'name' in line:
                if not 'data' in line:
                    if is_sub:
                        pos = line.rfind('_%s\"'%pos_fix)
                        line = line[:pos] + '\"'
                    else:
                        pos = line.rfind('\"')
                        line = line[:pos] + '_' + pos_fix +line[pos:]
            print >>fout,line

if net_param!="":
    net = caffe.Net(net_proto,net_param,caffe.TEST)
    net2 =  caffe.Net(net_new_proto,caffe.TEST)

    for name,param in net.params.items():
        print 'copy param '+name
        if is_usb:
            pos = name.rfind('_%s' % pos_fix)
            new_name = name[:pos]
        else:
            new_name =name+'_' + pos_fix
        for ix,sub_param  in enumerate(param):
            net2.params[new_name][ix].data[:] = sub_param.data.copy()

    base_name = os.path.basename(net_param).strip().split('.')
    net_new_param = base_name[0]+'_'+pos_fix+'.'+base_name[1]
    net2.save(net_new_param)
