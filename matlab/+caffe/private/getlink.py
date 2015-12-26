import glob
import os

cmd = 'mklink %s %s'
for file in glob.glob(r'D:\users\v-guoxie\caffe\3rdparty\bin\*.dll'):
    name = os.path.basename(file)
    c_cmd = cmd % (name, file)
    os.system(c_cmd)
