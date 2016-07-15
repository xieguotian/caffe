import base64
import os
import sys
import numpy as np

img_list = sys.argv[1]
img_folder = sys.argv[2]
b64_save_path = sys.argv[3]
with open(b64_save_path,'w') as fout:
    with open(img_list) as fid:
        for ix,line in enumerate(fid):
            if ((ix+1)%1000==0):
                print "process %d" %(ix+1)
            str = line.split()
            img_path = str[0].strip()
            label = np.int(str[1])

            with open(img_folder+'/'+img_path,'rb') as fimg:
                img_str = fimg.read()
                img_b64 = base64.b64encode(img_str)
                print >>fout,'%09d_%s\t%s\t%d'%(ix,img_path,img_b64,label)
