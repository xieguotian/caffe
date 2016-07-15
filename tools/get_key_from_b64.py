import sys
import numpy as np

b64_file = sys.argv[1]
save_file = sys.argv[2]
base_tmp = np.int(sys.argv[3])

key_set = set()
count = base_tmp - 1
with open(b64_file) as fid:
    with open(save_file,'w') as fout:
        for line in fid:
            key = line.split('\t')[0].strip()
            key_id = key.split('_')[0].strip()

            if key_id not in key_set:
                count+=1
                key_set.add(key_id)
            print >>fout,'%s %d'%(key,count)