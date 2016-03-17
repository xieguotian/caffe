import socket
from collections import OrderedDict
import numpy as np
localAdrr = '127.0.0.1'
port = 13000

data = ''

root_path = 'D:/users/v-guoxie/work_place/project/dog breed/data/clickture_dog_eval/'
img_list = []
#with open(root_path+'clickture_dog_eval_list.txt') as fid:
with open('clickture_dog_eval_list_3000.txt') as fid:
    for line in fid:
        img_list.append(line.strip().split()[0].strip())
print 'total images %d'%len(img_list)
tag_dict = OrderedDict()
with open('clickture_dog_name_label.txt') as fid:
    for line in fid:
        str = line.strip().split('\t')
        tag_dict[str[0].strip()] = np.int(str[1].strip())

with open('result_3000_3.txt','w') as fout:
    for ix,img_name in enumerate(img_list):
        #if ix <160598:
        #    continue
        sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        sock.connect((localAdrr,port))
        send_data = root_path+img_name
        sock.sendall(send_data)
        data = ''
        data = sock.recv(1024)
        if data:
            all_result = data.strip().split(';')
            all_label = []
            for result in all_result:
                str = result.strip().split(':')
                label = np.int(tag_dict[str[0].strip()])+1
                all_label.append(label)
            print >>fout,'%d %d %d %d %d'%(all_label[0],all_label[1],all_label[2],all_label[3],all_label[4])
        sock.close()
        if (ix+1)%100==0:
            print 'process %d'%(ix+1)