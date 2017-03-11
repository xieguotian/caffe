import os
import cPickle

import numpy as np
import sklearn
import sklearn.linear_model

import lmdb
import caffe

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
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
    cifar_python_directory = os.path.abspath("cifar-10-batches-py")

    #meta=unpickle(os.path.join(cifar_python_directory, 'meta'))
    #fine_label_names=meta['fine_label_names']
    #print(fine_label_names)

    print("Converting...")
    cifar_caffe_directory=os.path.abspath("cifar10_train_lmdb")
    if not os.path.exists(cifar_caffe_directory):

        X1,y_f1=load_data(os.path.join(cifar_python_directory, 'data_batch_1'))
        X2, y_f2 = load_data(os.path.join(cifar_python_directory, 'data_batch_2'))
        X3,y_f3 =load_data(os.path.join(cifar_python_directory, 'data_batch_3'))
        X4, y_f4 = load_data(os.path.join(cifar_python_directory, 'data_batch_4'))
        X5,y_f5=load_data(os.path.join(cifar_python_directory, 'data_batch_5'))

        X = np.concatenate((X1,X2,X3,X4,X5),axis=0)
        y_f = np.concatenate((y_f1,y_f2,y_f3,y_f4,y_f5),axis=0)
        #X,y_f = shuffle_data(data,y_f)
        idx = np.random.permutation(X.shape[0])
        X = X[idx,:,:,:]
        y_f = y_f[idx]

        Xt,yt_f=load_data(os.path.join(cifar_python_directory, 'test_batch'))

        print("Data is fully loaded,now truly convertung.")

        env=lmdb.open(cifar_caffe_directory,map_size=50000*1000*5)
        txn=env.begin(write=True)
        count=0
        for i in range(X.shape[0]):
            datum=caffe.io.array_to_datum(X[i],y_f[i])
            str_id='{:08}'.format(count)
            txn.put(str_id,datum.SerializeToString())

            count+=1
            if count%1000==0:
                print('already handled with {} pictures'.format(count))
                txn.commit()
                txn=env.begin(write=True)

        txn.commit()
        env.close()

        env=lmdb.open('cifar10_test_lmdb',map_size=10000*1000*5)
        txn=env.begin(write=True)
        count=0
        for i in range(Xt.shape[0]):
            datum=caffe.io.array_to_datum(Xt[i],yt_f[i])
            str_id='{:08}'.format(count)
            txn.put(str_id,datum.SerializeToString())

            count+=1
            if count%1000==0:
                print('already handled with {} pictures'.format(count))
                txn.commit()
                txn=env.begin(write=True)

        txn.commit()
        env.close()
    else:
        print("Conversion was already done. Did not convert twice.")