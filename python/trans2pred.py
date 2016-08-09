import numpy as np
import sys
import os
if not len(sys.argv)==3:
    print 'Usage:python trans2pred.py input_file output_text'

input_file = sys.argv[1]
output_text = sys.argv[2]

top_num = 5
result_mat = np.load(input_file)
result_idx = result_mat.argsort(axis=1)[:,::-1][:,:top_num]

txt_file = os.path.dirname(os.path.abspath(__file__)) + '\\synset_sort.txt'
sort_idx =np.loadtxt(txt_file)
result_idx = sort_idx[result_idx.flatten()].reshape(result_idx.shape)
np.savetxt(output_text,result_idx,fmt='%d')
