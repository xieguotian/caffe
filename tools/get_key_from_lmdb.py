import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2
import sys

db_name = sys.argv[1]
save_name = sys.argv[2]
re_label = None
if len(sys.argv)>=4:
    re_label = np.int(sys.argv[3])

save_prefix = db_name[:db_name.rfind('_')]
print db_name,save_prefix
lmdb_env = lmdb.open(db_name)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe_pb2.Datum()

with open(save_name,'w') as fout:
    for ix,(key, value) in enumerate(lmdb_cursor):
        datum.ParseFromString(value)
        if re_label==None:
            label = datum.label
        else:
            label = re_label
        #pos = key.find('_')
        #key = key[pos+1:]
        print >>fout,'%s %d'%(key.strip(),np.int(label))
        if(ix+1)%10000==0:
            print 'process %d'%(ix+1)