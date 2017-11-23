import os
import pickle

import numpy as np
import sklearn
import sklearn.linear_model

import lmdb
import caffe
import glob

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict

def shuffle_data(data, labels):
    data, _, labels, _ = sklearn.cross_validation.train_test_split(
        data, labels, test_size=0.0, random_state=42
    )
    return data, labels

def load_data(train_file):
    d = unpickle(train_file)
    data = d['data']
    #coarse_labels = d['coarse_labels']
    fine_labels = d['labels']
    length = len(d['labels'])

    data, labels = shuffle_data(
        data,
        #np.array(zip(coarse_labels, fine_labels))
        np.array(fine_labels)
    )
    #coarse_labels, fine_labels = zip(*labels.tolist())
    return (
        data.reshape(length, 3, 32, 32),
        #np.array(coarse_labels),
        labels
    )

if __name__=='__main__':
    data_path = 'Imagenet32_train'
    save_path = data_path+'_lmdb'
    all_files = glob.glob(data_path+'/*_p2')
    X,Y = load_data(all_files[0])
    for i in range(1,len(all_files)):
        X_tmp,Y_tmp= load_data(all_files[i])
        X = np.concatenate((X,X_tmp),axis=0)
        Y = np.concatenate((Y,Y_tmp),axis=0)

    idx = np.random.permutation(X.shape[0])
    X = X[idx,:,:,:]
    Y = Y[idx]

    print("Data is fully loaded,now truly convertung.")
    map_size = X.nbytes * 1.5
    env=lmdb.open(save_path,map_size)
    txn=env.begin(write=True)
    count=0
    for i in range(X.shape[0]):
        datum=caffe.io.array_to_datum(X[i],Y[i])
        str_id='{:09}'.format(count)
        txn.put(str_id,datum.SerializeToString())

        count+=1
        if count%1000==0:
            print('already handled with {} pictures'.format(count))
            txn.commit()
            txn=env.begin(write=True)

    txn.commit()
    env.close()