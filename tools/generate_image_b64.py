import base64
import os
import sys
import numpy as np
import argparse
from cv_util.image_util import resize_image_preserve_ratio
import cv2

parser = argparse.ArgumentParser()
parser.add_argument(
    "image_list",
    help="a file that is a list of image name"
)
parser.add_argument(
    "image_folder",
    help="a folder contain images"
)
parser.add_argument(
    "b64_save_path",
    help="path to save the base64 format db."
)
parser.add_argument(
    "--resize_scale",
    type=int,
    default=0,
    help="the smallest scale to save the image. it preserve ratio"
)
args = parser.parse_args()

img_list = args.image_list
img_folder = args.image_folder
b64_save_path = args.b64_save_path
resize_scale = args.resize_scale

with open(b64_save_path,'w') as fout:
    with open(img_list) as fid:
        for ix,line in enumerate(fid):
            if ((ix+1)%1000==0):
                print "process %d" %(ix+1)
            str = line.split()
            img_path = str[0].strip()
            label = np.int(str[1])
            if resize_scale>0:
                img = cv2.imread(img_folder+'/'+img_path)
                img = resize_image_preserve_ratio(img,resize_scale)
                img_str = cv2.imencode(os.path.splitext(img_path)[1],img)[1].tostring()
                img_b64 = base64.b64encode(img_str)
                print >>fout,'%09d_%s\t%s\t%d'%(ix,img_path,img_b64,label)
            else:
                with open(img_folder+'/'+img_path,'rb') as fimg:
                    img_str = fimg.read()
                    img_b64 = base64.b64encode(img_str)
                    print >>fout,'%09d_%s\t%s\t%d'%(ix,img_path,img_b64,label)
