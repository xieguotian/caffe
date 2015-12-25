#!/usr/bin/env python
"""
classify.py is an out-of-the-box image classifer callable from the command line.

By default it configures and runs the Caffe reference ImageNet model.
"""
import numpy as np
import os
import sys
import argparse
import glob
import time

import caffe


def main(argv):
    pycaffe_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "input_file",
        help="Input image, directory, or npy."
    )
    parser.add_argument(
        "output_file",
        help="Output npy filename."
    )
    # Optional arguments.
    parser.add_argument(
        "--model_def",
        default=os.path.join(pycaffe_dir,
                "../models/bvlc_reference_caffenet/deploy.prototxt"),
        help="Model definition file."
    )
    parser.add_argument(
        "--pretrained_model",
        default=os.path.join(pycaffe_dir,
                "../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"),
        help="Trained model weights file."
    )
    parser.add_argument(
        "--gpu",
        action='store_true',
        help="Switch for gpu computation."
    )
    parser.add_argument(
        "--center_only",
        action='store_true',
        help="Switch for prediction from center crop alone instead of " +
             "averaging predictions across crops (default)."
    )
    parser.add_argument(
        "--images_dim",
        default='256,256',
        help="Canonical 'height,width' dimensions of input images."
    )
    parser.add_argument(
        "--mean_file",
        default=os.path.join(pycaffe_dir,
                             'caffe/imagenet/ilsvrc_2012_mean.npy'),
        help="Data set image mean of [Channels x Height x Width] dimensions " +
             "(numpy array). Set to '' for no mean subtraction."
    )
    parser.add_argument(
        "--input_scale",
        type=float,
        help="Multiply input features by this scale to finish preprocessing."
    )
    parser.add_argument(
        "--raw_scale",
        type=float,
        default=255.0,
        help="Multiply raw input by this scale before preprocessing."
    )
    parser.add_argument(
        "--channel_swap",
        default='2,1,0',
        help="Order to permute input channels. The default converts " +
             "RGB -> BGR since BGR is the Caffe default by way of OpenCV."
    )
    parser.add_argument(
        "--ext",
        default='jpg',
        help="Image file extension to take as input when a directory " +
             "is given as the input file."
    )

    parser.add_argument(
        "--nimg_per_iter",
        type=int,
        default=100,
        help="number of images to be processd per iteration."
    )

    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="device id of GPU"
    )
    parser.add_argument(
        "--root_path",
        default="./",
        help="root path of image if input_file is a txt list."
    )
    parser.add_argument(
        "--pre_fetch",
        action='store_true',
        help="pre-fetech image data with a new thread."
    )
    parser.add_argument(
        "--not_resize_image",
        action='store_true',
        help='not to resize image, only crop from the origin image.'
    )
    parser.add_argument(
        "--dense",
        action='store_true',
        help='use dense prediction'
    )
    parser.add_argument(
        "--preserve_ratio",
        action='store_true',
        help='presrve_ratio when resize image'
    )
    args = parser.parse_args()

    image_dims = [int(s) for s in args.images_dim.split(',')]

    mean, channel_swap = None, None
    if args.mean_file:
        mean = np.load(args.mean_file).mean(1).mean(1)
    if args.channel_swap:
        channel_swap = [int(s) for s in args.channel_swap.split(',')]

    if args.gpu:
        caffe.set_mode_gpu()
        caffe.set_device(args.device_id)
        print("GPU mode")
    else:
        caffe.set_mode_cpu()
        print("CPU mode")

    if args.dense:
        classifier = caffe.Classifier_dense(args.model_def, args.pretrained_model,
                    image_dims=image_dims, mean=mean,
                    input_scale=args.input_scale, raw_scale=args.raw_scale,
                    channel_swap=channel_swap,preserve_ratio=args.preserve_ratio)
    else:
        if args.pre_fetch:
            # Make classifier.
            if args.not_resize_image:
                resize_img = False
            else:
                resize_img = True
            classifier = caffe.Classifier_parallel(args.model_def, args.pretrained_model,
                    image_dims=image_dims, mean=mean,
                    input_scale=args.input_scale, raw_scale=args.raw_scale,resize_image=not args.not_resize_image,
                    channel_swap=channel_swap,preserve_ratio=args.preserve_ratio)
        else:
            # Make classifier.
            classifier = caffe.Classifier(args.model_def, args.pretrained_model,
                    image_dims=image_dims, mean=mean,
                    input_scale=args.input_scale, raw_scale=args.raw_scale,
                    channel_swap=channel_swap)

    # Load numpy array (.npy), directory glob (*.jpg), or image file.
    args.input_file = os.path.expanduser(args.input_file)
    if args.input_file.endswith('npy'):
        print("Loading file: %s" % args.input_file)
        inputs = np.load(args.input_file)
    elif args.input_file.endswith('txt'):
        print("Loading txt file: %s" % args.input_file)
        fid = open(args.input_file,'r')
        list = [args.root_path+'/'+name[:-1] for name in fid]
        inputs = list
    elif os.path.isdir(args.input_file):
        print("Loading folder: %s" % args.input_file)
        #inputs =[caffe.io.load_image(im_f)
        #         for im_f in glob.glob(args.input_file + '/*.' + args.ext)]
        list = glob.glob(args.input_file + '/*.' + args.ext)
        inputs = list
    else:
        print("Loading file: %s" % args.input_file)
        inputs = [caffe.io.load_image(args.input_file)]

    print("Classifying %d inputs." % len(inputs))

    # Classify.
    start = time.time()
    if len(list)!=0:
        if args.dense:
            n_p_iter = args.nimg_per_iter
            predictions = classifier.predict(list,nimg_per_iter=n_p_iter)
        else:
            if args.pre_fetch:
                n_p_iter = args.nimg_per_iter
                predictions = classifier.predict(list,oversample=not args.center_only,nimg_per_iter=n_p_iter)
            else:
                predictions = []
                total_num_images = len(list)
                n_p_iter = args.nimg_per_iter
                for idx in range(0,total_num_images,n_p_iter):
                    end_idx = min(total_num_images,idx+n_p_iter)
                    inputs = [caffe.io.load_image(im_f)
                            for im_f in list[idx:end_idx]]
                    predictions.extend(classifier.predict(inputs, not args.center_only))
                    print 'process {} images'.format(end_idx)
    else:
        predictions = classifier.predict(inputs, not args.center_only)
    print("Done in %.2f s." % (time.time() - start))

    # Save
    print("Saving results into %s" % args.output_file)
    np.save(args.output_file, predictions)


if __name__ == '__main__':
    main(sys.argv)
