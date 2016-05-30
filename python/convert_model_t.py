import caffe
import sys
import os

if len(sys.argv)<4:
    print 'Usage: python convert_model.py net_proto net_param net_posfix'
net_proto = sys.argv[1]
net_param = sys.argv[2]
pos_fix = sys.argv[3]

base_name = os.path.basename(net_proto).strip().split('.')
net_new_proto = base_name[0]+'_'+pos_fix+'.'+base_name[1]

with open(net_proto) as fid:
    with open(net_new_proto,'w') as fout:
        for line in fid:
            line = line.strip('\n')
            if 'bottom' in line or 'top' in line or 'name' in line:
                if not 'data' in line:
                    pos = line.rfind('\"')
                    line = line[:pos] + '_' + pos_fix +line[pos:]
            print >>fout,line

net = caffe.Net(net_proto,net_param,caffe.TEST)
net2 =  caffe.Net(net_new_proto,caffe.TEST)

for name,param in net.params.items():
    print 'copy param '+name
    new_name =name+'_' + pos_fix
    for ix,sub_param  in enumerate(param):
        net2.params[new_name][ix].data[:] = sub_param.data.copy()

base_name = os.path.basename(net_param).strip().split('.')
net_new_param = base_name[0]+'_'+pos_fix+'.'+base_name[1]
net2.save(net_new_param)
