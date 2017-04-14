import numpy as np
import sys
import os
import glob
import fnmatch
#argv[0]:base path
#argv[1]:save path
#argv[2]:postfix of files

def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename

argc = len(sys.argv)
if not argc==4:
	print 'Usage: python get_list.py base_path save_file ext'

base_path = sys.argv[1]
save_file = sys.argv[2]
ext = sys.argv[3]

#list = glob.glob(base_path + '/*.' + ext)

fid = open(save_file,'w')
for name in find_files(base_path,'*.'+ext):#list:
	#name = os.path.basename(name)
	label = os.path.basename(os.path.dirname(name))
	print >> fid,'%s\t%d'%(name,np.int(label))
fid.close()