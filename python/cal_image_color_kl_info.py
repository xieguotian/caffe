import cv2
import numpy as np
import os
import sys
import sklearn

file_list = sys.argv[1]
file_path = sys.argv[2]
save_name = sys.argv[3]

with open(file_list) as fid:
    with open(save_name,'w') as fout:
        for ix,line in enumerate(fid):
            f_path = file_path+'/'+line[9:].split()[0].strip()
            img = cv2.imread(f_path,cv2.IMREAD_COLOR)
            tmp_mat = img.reshape(img.shape[0]*img.shape[1],img.shape[2]).transpose()
            tmp_cov = np.cov(tmp_mat)
            U,s,V = np.linalg.svd(tmp_cov)
            sqrtV = np.sqrt(s)

            result = line.strip()
            for value in U.flatten():
                result = result + ' %0.2f'%(value)
            for value in sqrtV.flatten():
                result = result + ' %0.2f' %(value)
            print >>fout,result
            if (ix+1)%1000==0:
                print 'process %d'%(ix+1)



