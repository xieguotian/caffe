#!/usr/bin/env python
"""
Classifier is an image classifier specialization of Net.
"""

import numpy as np
from multiprocessing import Process, Queue, cpu_count

import caffe
import time
from PIL import Image
import matplotlib.pyplot as plt
import sys

class Classifier(caffe.Net):
    """
    Classifier extends Net for image class prediction
    by scaling, center cropping, or oversampling.

    Parameters
    ----------
    image_dims : dimensions to scale input for cropping/sampling.
        Default is to scale to net input size for whole-image crop.
    mean, input_scale, raw_scale, channel_swap: params for
        preprocessing options.
    """
    def __init__(self, model_file, pretrained_file, image_dims=None,
                 mean=None, input_scale=None, raw_scale=None,
                 channel_swap=None):
        caffe.Net.__init__(self, model_file, pretrained_file, caffe.TEST)

        # configure pre-processing
        in_ = self.inputs[0]
        self.transformer = caffe.io.Transformer(
            {in_: self.blobs[in_].data.shape})
        self.transformer.set_transpose(in_, (2, 0, 1))
        if mean is not None:
            self.transformer.set_mean(in_, mean)
        if input_scale is not None:
            self.transformer.set_input_scale(in_, input_scale)
        if raw_scale is not None:
            self.transformer.set_raw_scale(in_, raw_scale)
        if channel_swap is not None:
            self.transformer.set_channel_swap(in_, channel_swap)

        self.crop_dims = np.array(self.blobs[in_].data.shape[2:])
        if not image_dims:
            image_dims = self.crop_dims
        self.image_dims = image_dims

    def predict(self, inputs, oversample=True):
        """
        Predict classification probabilities of inputs.

        Parameters
        ----------
        inputs : iterable of (H x W x K) input ndarrays.
        oversample : boolean
            average predictions across center, corners, and mirrors
            when True (default). Center-only prediction when False.

        Returns
        -------
        predictions: (N x C) ndarray of class probabilities for N images and C
            classes.
        """
        # Scale to standardize input dimensions.
        input_ = np.zeros((len(inputs),
                           self.image_dims[0],
                           self.image_dims[1],
                           inputs[0].shape[2]),
                          dtype=np.float32)
        for ix, in_ in enumerate(inputs):
            input_[ix] = caffe.io.resize_image(in_, self.image_dims)

        if oversample:
            # Generate center, corner, and mirrored crops.
            input_ = caffe.io.oversample(input_, self.crop_dims)
        else:
            # Take center crop.
            center = np.array(self.image_dims) / 2.0
            crop = np.tile(center, (1, 2))[0] + np.concatenate([
                -self.crop_dims / 2.0,
                self.crop_dims / 2.0
            ])
            input_ = input_[:, crop[0]:crop[2], crop[1]:crop[3], :]

        # Classify
        caffe_in = np.zeros(np.array(input_.shape)[[0, 3, 1, 2]],
                            dtype=np.float32)
        for ix, in_ in enumerate(input_):
            caffe_in[ix] = self.transformer.preprocess(self.inputs[0], in_)
        out = self.forward_all(**{self.inputs[0]: caffe_in})
        predictions = out[self.outputs[0]]

        # For oversampling, average predictions across crops.
        if oversample:
            predictions = predictions.reshape((len(predictions) / 10, 10, -1))
            predictions = predictions.mean(1)

        return predictions

def resize_img(img,img_dims,interp_order=Image.BICUBIC):
    tmp_img = Image.fromarray(np.uint8(img*255))
    tmp_img = tmp_img.resize(img_dims,interp_order)
    return np.array(tmp_img)/255.0

class Image_Data(Process):
    def __init__(self,img_list,image_dims,crop_dims,transformer,inputs,resize_image=True,
                 oversample=True,nimg_per_iter=200,queue_size=10,preserve_ratio=False):
        Process.__init__(self)

        self.oversample=oversample
        self.resize_image=resize_image
        self.queue_size=queue_size
        self.n_p_iter = nimg_per_iter
        self.img_list = img_list

        self.image_dims = image_dims
        self.crop_dims = crop_dims
        self.transformer = transformer
        self.inputs = inputs
        self.preserve_ratio = preserve_ratio

        self.img_queue = Queue(self.queue_size)

    def run(self):
        total_num_images = len(self.img_list)
        n_p_iter = self.n_p_iter
        for idx in range(0,total_num_images,n_p_iter):
            end_idx = min(total_num_images,idx+n_p_iter)
            #read images
            inputs = [caffe.io.load_image(im_f)
                    for im_f in self.img_list[idx:end_idx]]

            if not self.preserve_ratio:
                input_ = np.zeros((len(inputs),
                           self.image_dims[0],
                           self.image_dims[1],
                           inputs[0].shape[2]),
                          dtype=np.float32)
                #resize images
                for ix,in_ in enumerate(inputs):
                    #input_[ix] = caffe.io.resize_image(in_, self.image_dims,interp_order=3)
                    if not self.resize_image:
                        input_[ix] = in_
                    else:
                        input_[ix] = resize_img(in_,self.image_dims,Image.BICUBIC)
                #crop image
                if self.oversample:
                    # Generate center, corner, and mirrored crops.
                    input_ = caffe.io.oversample(input_, self.crop_dims)
                else:
                    # Take center crop.
                    center = np.array(self.image_dims) / 2.0
                    crop = np.tile(center, (1, 2))[0] + np.concatenate([
                        -self.crop_dims / 2.0,
                        self.crop_dims / 2.0
                    ])
                    input_ = input_[:, crop[0]:crop[2], crop[1]:crop[3], :]
            else:
                #crop images
                if self.oversample:
                    # oversample 10 views,Generate center, corner, and mirrored crops.
                    input_ = np.zeros((len(inputs)*10,
                                       self.crop_dims[0],
                                       self.crop_dims[1],
                                       inputs[0].shape[2]),
                                      dtype=np.float32)

                    for ix,in_ in enumerate(inputs):
                        # resize by preserving image ratio
                        if not self.resize_image:
                            tmp_in_ = in_
                        else:
                            height,width = in_.shape[:2]
                            if height<width:
                                ratio = self.image_dims[0] / float(height)
                                height = self.image_dims[0]
                                width = int(ratio*width)
                            elif width<height:
                                ratio = self.image_dims[0] / float(width)
                                height = int(ratio*height)
                                width = self.image_dims[0]
                            else:
                                height = self.image_dims[0]
                                width = self.image_dims[0]
                            #tmp_in_ = caffe.io.resize_image(in_, [height,width],interp_order=3)
                            tmp_in_ = resize_img(in_,[width,height],Image.BICUBIC)

                        input_[ix*10:(ix+1)*10] = caffe.io.oversample(tmp_in_[np.newaxis,:,:,:],self.crop_dims)
                else:
                    # center only
                    input_ = np.zeros((len(inputs),
                                       self.crop_dims[0],
                                       self.crop_dims[1],
                                       inputs[0].shape[2]),
                                      dtype=np.float32)

                    for ix,in_ in enumerate(inputs):
                        # resize by preserving image ratio
                        if not self.resize_image:
                            tmp_in_ = in_
                        else:
                            height,width = in_.shape[:2]
                            if height<width:
                                ratio = self.image_dims[0] / float(height)
                                height = self.image_dims[0]
                                width = int(ratio*width)
                            elif width<height:
                                ratio = self.image_dims[0] / float(width)
                                height = int(ratio*height)
                                width = self.image_dims[0]
                            else:
                                height = self.image_dims[0]
                                width = self.image_dims[0]

                            #tmp_in_ = caffe.io.resize_image(in_, [height,width],interp_order=3)
                            tmp_in_ = resize_img(in_,[width,height],Image.BICUBIC)
                        # Take center crop.
                        center = np.array(tmp_in_.shape[:2]) / 2.0
                        crop = np.tile(center, (1, 2))[0] + np.concatenate([
                            -self.crop_dims / 2.0,
                            self.crop_dims / 2.0
                            ])
                        input_[ix] = tmp_in_[crop[0]:crop[2], crop[1]:crop[3], :]

            # preprocess
            caffe_in = np.zeros(np.array(input_.shape)[[0, 3, 1, 2]],
                        dtype=np.float32)
            for ix, in_ in enumerate(input_):
                caffe_in[ix] = self.transformer.preprocess(self.inputs[0], in_)

            self.img_queue.put(caffe_in)

class Classifier_parallel(caffe.Net):
    """
    Classifier extends Net for image class prediction
    by scaling, center cropping, or oversampling.

    Parameters
    ----------
    image_dims : dimensions to scale input for cropping/sampling.
        Default is to scale to net input size for whole-image crop.
    mean, input_scale, raw_scale, channel_swap: params for
        preprocessing options.
    """
    def __init__(self, model_file, pretrained_file, image_dims=None,
                 mean=None, input_scale=None, raw_scale=None,resize_image=True,
                 channel_swap=None,preserve_ratio=False):
        caffe.Net.__init__(self, model_file, pretrained_file, caffe.TEST)

        # configure pre-processing
        in_ = self.inputs[0]
        self.transformer = caffe.io.Transformer(
            {in_: self.blobs[in_].data.shape})
        self.transformer.set_transpose(in_, (2, 0, 1))
        if mean is not None:
            self.transformer.set_mean(in_, mean)
        if input_scale is not None:
            self.transformer.set_input_scale(in_, input_scale)
        if raw_scale is not None:
            self.transformer.set_raw_scale(in_, raw_scale)
        if channel_swap is not None:
            self.transformer.set_channel_swap(in_, channel_swap)

        self.resize_image = resize_image

        self.crop_dims = np.array(self.blobs[in_].data.shape[2:])
        if not image_dims:
            image_dims = self.crop_dims
        self.image_dims = image_dims
        self.preserve_ratio = preserve_ratio
    def predict(self, img_list,nimg_per_iter=100, oversample=True):
        """
        Predict classification probabilities of inputs.

        Parameters
        ----------
        inputs : iterable of (H x W x K) input ndarrays.
        oversample : boolean
            average predictions across center, corners, and mirrors
            when True (default). Center-only prediction when False.

        Returns
        -------
        predictions: (N x C) ndarray of class probabilities for N images and C
            classes.
        """
        # Scale to standardize input dimensions.
        num_cores = min(max(cpu_count()-1,1),10)
        fetch_sets = []
        step = num_cores*nimg_per_iter
        for i in range(num_cores):
            tmp_list = []
            for idx in range(i*nimg_per_iter,len(img_list),step):
                end_idx = min(idx+nimg_per_iter,len(img_list))
                tmp_list.extend(img_list[idx:idx+nimg_per_iter])
            print i, len(tmp_list),step
            fetch_sets.append(Image_Data(tmp_list,self.image_dims,self.crop_dims,self.transformer,self.inputs
                                         ,self.resize_image,oversample,nimg_per_iter,queue_size=3,preserve_ratio=self.preserve_ratio))
            fetch_sets[i].start()

        print 'start thread to read images'

        predictions = []
        count = 0
        total_num_images = len(img_list)
        n_p_iter = nimg_per_iter
        for ix,idx in enumerate(range(0,total_num_images,n_p_iter)):
            # get thread for data
            th_idx = ix % num_cores
            print th_idx
            image_data_prefetch = fetch_sets[th_idx]

            # show time for fetch
            flag = False
            if image_data_prefetch.img_queue.empty():
                time_st = time.clock()
                flag=True
                print 'wait for prefetch image data'
            # get data
            caffe_in = image_data_prefetch.img_queue.get()
            if flag:
                print 'time for prefetch: %fs' % (time.clock()-time_st)

            # classify
            time_st = time.clock()
            out = self.forward_all(**{self.inputs[0]: caffe_in})
            print 'time for predition: %fs' % (time.clock()-time_st)


            predictions_ = out[self.outputs[0]]

            # For oversampling, average predictions across crops.
            if oversample:
                predictions_ = predictions_.reshape((len(predictions_) / 10, 10, -1))
                predictions_ = predictions_.mean(1)

            predictions.extend(predictions_)

            if oversample:
                count += caffe_in.shape[0] / 10.0
            else:
                count += caffe_in.shape[0]
            print '%d images processed' % count
        return predictions

def preprocess(img,data_mean,raw_scale,input_scale):
    if raw_scale is not None:
        img *= raw_scale
    img = np.float32(np.rollaxis(img, 2)[::-1]) - (data_mean)
    if input_scale is not None:
        img *= input_scale
    return img
def deprocess(img,mean):
    return np.dstack((img + mean)[::-1])

def process_images(img_queue,img_list,data_mean,img_dims,preserve_ratio,raw_scale=None,input_scale=None):
    for img_name in img_list:
        img = caffe.io.load_image(img_name)
        if not preserve_ratio:
            #img = caffe.io.resize_image(img,img_dims,interp_order=3)
            img = resize_img(img,img_dims,Image.BICUBIC)
        else:
            height,width = img.shape[:2]
            if height<width:
                ratio = img_dims[0] / float(height)
                height = img_dims[0]
                width = int(ratio*width)
            elif width < height:
                ratio = img_dims[0] / float(width)
                height = int(ratio*height)
                width = img_dims[0]
            else:
                height = img_dims[0]
                width = img_dims[0]

            #img = caffe.io.resize_image(img,[height,width],interp_order=3)
            img = resize_img(img,[width,height],Image.BICUBIC)

        caffe_in = preprocess(img,data_mean,raw_scale,input_scale)
        caffe_in = caffe_in[np.newaxis,:,:,:]
        img_queue.put(caffe_in)

class Classifier_dense(caffe.Net):
    """
    Classifier extends Net for image class prediction
    by scaling, center cropping, or oversampling.

    Parameters
    ----------
    image_dims : dimensions to scale input for cropping/sampling.
        Default is to scale to net input size for whole-image crop.
    mean, input_scale, raw_scale, channel_swap: params for
        preprocessing options.
    """
    def __init__(self, model_file, pretrained_file, image_dims=None,
                 mean=None, input_scale=None, raw_scale=None,
                 channel_swap=None,preserve_ratio=False):
        caffe.Net.__init__(self, model_file, pretrained_file, caffe.TEST)

        # configure pre-processing
        in_ = self.inputs[0]
        self.transformer = caffe.io.Transformer(
            {in_: self.blobs[in_].data.shape})
        self.transformer.set_transpose(in_, (2, 0, 1))
        if mean is not None:
            self.transformer.set_mean(in_, mean)
        if input_scale is not None:
            self.transformer.set_input_scale(in_, input_scale)
        if raw_scale is not None:
            self.transformer.set_raw_scale(in_, raw_scale)
        if channel_swap is not None:
            self.transformer.set_channel_swap(in_, channel_swap)

        self.crop_dims = np.array(self.blobs[in_].data.shape[2:])
        if not image_dims:
            image_dims = self.crop_dims
        self.image_dims = image_dims
        self.preserve_ratio = preserve_ratio

    def predict(self, img_list,nimg_per_iter=100):
        predictions = []
        img_queue = Queue(20)
        img_processor = Process(target=process_images,
                                args=(img_queue,
                                      img_list,
                                      self.transformer.mean.get(self.inputs[0]),
                                      self.image_dims,
                                      self.preserve_ratio,
                                      self.transformer.raw_scale.get(self.inputs[0]),
                                      self.transformer.input_scale.get(self.inputs[0])))
        img_processor.start()
        time_pred_st = time.clock()
        for idx,img_name in enumerate(img_list):
            '''
            flag = False
            if img_queue.empty():
                time_st = time.clock()
                flag = True
                print 'image queue is empty'
            '''

            caffe_in = img_queue.get()
            '''
            if flag:
                print 'time for pre_read images: %fs' % (time.clock()-time_st)
            '''
            self.blobs[self.inputs[0]].reshape(caffe_in.shape[0],
                                               caffe_in.shape[1],
                                               caffe_in.shape[2],
                                               caffe_in.shape[3])
            out = self.forward_all(**{self.inputs[0]: caffe_in})

            if idx % nimg_per_iter == 0:
                print '%d process,time for predicting %d images: %fs'%(idx,nimg_per_iter,time.clock()-time_pred_st)
                time_pred_st = time.clock()

            prediction = out[self.outputs[0]].mean(2).mean(2)
            predictions.extend(prediction)

        if not img_queue.empty():
            print 'some images left.'
        if img_processor.is_alive():
            img_processor.terminate()
        return predictions

