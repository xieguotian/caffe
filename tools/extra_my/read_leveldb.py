import caffe
import leveldb
import lmdb
import cPickle
import numpy as np
from pylearn2.utils import string_utils

train =  'cifar100_pre_train_lmdb'
test =  'cifar100_pre_test_lmdb'
#db = leveldb.LevelDB(db_name)

train_data_dir = string_utils.preprocess('${PYLEARN2_DATA_PATH}/cifar100')+'/pylearn2_gcn_whitened/train.pkl'
test_data_dir = string_utils.preprocess('${PYLEARN2_DATA_PATH}/cifar100')+'/pylearn2_gcn_whitened/test.pkl'
with open(train_data_dir) as train_fid:
    with open(test_data_dir) as test_fid:
        data = cPickle.load(train_fid)
        train_img_data = data.get_data()[0]
        train_img_data = train_img_data.reshape(train_img_data.shape[0], 3, 32, 32)
        train_label = data.get_data()[1]

        data = cPickle.load(test_fid)
        test_img_data = data.get_data()[0]
        test_img_data = test_img_data.reshape(test_img_data.shape[0], 3, 32, 32)
        test_label = data.get_data()[1]

        min_value = min(train_img_data.min(),test_img_data.min())
        train_img_data = train_img_data - min_value
        test_img_data = test_img_data - min_value
        max_value = max(train_img_data.max(),test_img_data.max())
        scale = 255.0 / max_value

        train_img_data = scale*train_img_data
        test_img_data = scale * test_img_data

        train_img_data[train_img_data>255] = 255
        train_img_data[train_img_data<0] = 0
        test_img_data[test_img_data>255] = 255
        test_img_data[test_img_data<0] = 0

        train_img_data = train_img_data.astype(np.uint8)
        test_img_data = test_img_data.astype(np.uint8)

        print 'scale:%f,%f,min:%f,max:%f'%(scale,1.0/scale,min_value,max_value)

        train_env=lmdb.open(train,map_size=50000*1000*5)
        train_txn=train_env.begin(write=True)
        count = 0

        for i in range(train_img_data.shape[0]):
            datum=caffe.io.array_to_datum(train_img_data[i],np.int(train_label[i]))
            str_id='{:08}'.format(count)
            train_txn.put(str_id,datum.SerializeToString())
            count+=1
            if count%1000==0:
                print('already handled with {} pictures'.format(count))
                train_txn.commit()
                train_txn=train_env.begin(write=True)

        train_txn.commit()
        train_env.close()

        test_env=lmdb.open(test,map_size=10000*1000*5)
        test_txn=test_env.begin(write=True)
        count = 0

        for i in range(test_img_data.shape[0]):
            datum=caffe.io.array_to_datum(test_img_data[i],np.int(test_label[i]))
            str_id='{:08}'.format(count)
            test_txn.put(str_id,datum.SerializeToString())
            count+=1
            if count%1000==0:
                print('already handled with {} pictures'.format(count))
                test_txn.commit()
                test_txn=test_env.begin(write=True)
        test_txn.commit()
        test_env.close()

"""
with open(data_dir) as fid:
    data = cPickle.load(fid)
    img_data = data.get_data()[0]
    label = data.get_data()[1]
    img_data = img_data.reshape(img_data.shape[0], 3, 32, 32)

    min_value = -11.074004#img_data.min()
    print img_data.min()
    img_data = img_data-min_value
    max_value = 23.925400#img_data.max()
    print img_data.max()
    scale = 255.0 / max_value

    #scale = 10.658129 #10.658129,0.093825,min:-11.074004,max:23.925400
    #scale = 19.842204
    img_data = scale*img_data
    img_data[img_data>255] = 255
    img_data[img_data<0] = 0

    img_data = img_data.astype(np.uint8)
    #exit()
    print 'scale:%f,%f,min:%f,max:%f'%(scale,1.0/scale,min_value,max_value)
    for i in range(img_data.shape[0]):
        datum=caffe.io.array_to_datum(img_data[i],np.int(label[i]))
        str_id='{:08}'.format(count)
        txn.put(str_id,datum.SerializeToString())
        count+=1
        if count%1000==0:
            print('already handled with {} pictures'.format(count))
            txn.commit()
            txn=env.begin(write=True)

txn.commit()
env.close()
"""