import sys
import os
import glob

#argv[0]:base path
#argv[1]:save path
#argv[2]:postfix of files

argc = len(sys.argv)
if not argc==4:
	print 'Usage: python get_list.py base_path save_file ext'

base_path = sys.argv[1]
save_file = sys.argv[2]
ext = sys.argv[3]

list = glob.glob(base_path + '/*.' + ext)

fid = open(save_file,'w')
for name in list:
	name = os.path.basename(name)
	print >> fid,name
fid.close()