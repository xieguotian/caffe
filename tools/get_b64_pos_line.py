import os
import sys
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "b64_file_name",
    help="base64 format database"
)
parser.add_argument(
    "--save_name",
    type=str,
    default="",
    help="save file name of the key-label-position tuple."
)
args = parser.parse_args()

file_name = args.b64_file_name
save_name = args.save_name
if save_name=="":
    save_name = os.path.splitext(os.path.basename(file_name))[0] + "_key_file.txt"

pos_arr = []
with open(file_name) as fid:
    with open(save_name,'w') as fout:
        count = 0
        while(True):
            pos = fid.tell()
            line=fid.readline()
            if (not line):
                break
            str = line.split('\t')
            key = str[0].strip()
            if len(str)>=3:
                label = np.int(str[2])
            else:
                str_tmp  = str[1].split()
                if len(str_tmp)>=2:
                    label = np.int(str[1])
                else:
                    label = 0
            count+=1
            if (count)%1000==0:
                print "process %d"%(count)
            print >> fout, '%s\t%d\t%d'%(key,label,pos)

