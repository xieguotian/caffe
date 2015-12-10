import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import time

import caffe


def showarray(a, fmt='png',saved=''):
    a = np.uint8(np.clip(a, 0, 255))
    img = PIL.Image.fromarray(a)
    plt.imshow(img)
    plt.show(block=False)
    #fig.set_data(img)
    plt.draw()
    if not saved == '':
        img.save(saved,fmt)

def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

def objective_L2(dst):
    dst.diff[:] = dst.data

def make_step(net, end, step_size=1.5, regular_size=0.005, clip=True, objective=objective_L2):
    '''Basic gradient ascent step.'''
    '''layer_name and top_name is consistent'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    net.forward(end=end)
    objective(dst)  # specify the optimization objective
    net.backward(start=end)

    g = src.diff[0]

    # apply normalized ascent step to the input image
    if not np.abs(g).mean()==0:
        src.data[:] += step_size/np.abs(g).mean() * g
        src.data[:] -= regular_size*step_size*src.data[:]

    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)


def make_step_by_layer(net, end, st, layer, step_size=1.5, regular_size=0.005, clip=True, objective=objective_L2):
    '''Basic gradient ascent step.'''
    '''layer_name and top_name is consistent'''

    #src = net.blobs['data'] # input image is stored in Net's 'data' blob
    src = net.blobs[st]
    dst = net.blobs[end]

    net.forward(start=layer,end=end)
    objective(dst)  # specify the optimization objective
    net.backward(start=end)

    g = src.diff[0]

    # apply normalized ascent step to the input image
    if not np.abs(g).mean()==0:
        src.data[:] += step_size/np.abs(g).mean() * g
        src.data[:] -= regular_size*step_size*src.data[:]

    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)


def feat_vis(net, base_img, end,iter_n=10, clip=True,show_vis=True, **step_params):
    # prepare base images for all octaves
    img = preprocess(net, base_img)

    src = net.blobs['data']
    h, w = img.shape[-2:]

    src.reshape(1,3,h,w) # resize the network's input image size
    src.data[0] = img

    for i in range(iter_n):
        make_step(net, end=end, clip=clip, **step_params)
        # visualization
        vis = deprocess(net, src.data[0])
        if not clip: # adjust image contrast if clipping is disabled
            vis = vis*(255.0/np.percentile(vis, 99.98))
        if show_vis:
            showarray(vis)
        print i, end, vis.shape
            #clear_output(wait=True)

    # returning the resulting image
    return deprocess(net, src.data[0])

def topdown_vis(net,feat,layer_list,blob_list,iter_n=10,clip=True,show_vis=True,**step_params):
    target = feat.copy()

    for end,start,layer in zip(blob_list[:-1],blob_list[1:],layer_list):
        src = net.blobs[start]
        print start,end,layer
        for i in range(iter_n):
            #objective function
            def objective_ED(dst):
                dst.diff[:] = target - dst.data
                #print dst.diff.mean()

            if start=='data':
                clip_tmp=clip
            else:
                clip_tmp=False

            #begin step
            make_step_by_layer(net,end,start,layer,clip=clip_tmp,objective=objective_ED)

            #show reconstruction image
            if start=='data' and show_vis:
                vis = deprocess(net,src.data[0])
                showarray(vis)
                print i,start,end,layer,vis.shape
        # update target
        target = src.data.copy()

    in_src = net.blobs[blob_list[-1]]
    return deprocess(net,in_src.data[0])

'''deep dream step'''
def make_step2(net, step_size=1.5, end='inception_4c/output',
              jitter=32, clip=True, objective=objective_L2):
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    if jitter==0:
        ox=0
        oy=0
    else:
        ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift

    net.forward(end=end)
    objective(dst)  # specify the optimization objective
    net.backward(start=end)
    g = src.diff[0]
    # apply normalized ascent step to the input image
    if not np.abs(g).mean()==0:
        src.data[:] += step_size/np.abs(g).mean() * g
        src.data[:] -= 0.005*step_size*src.data[:]

    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image

    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)

def deepdream(net, base_img, iter_n=10, octave_n=4, octave_scale=1.4,
              end='inception_4c/output', clip=True, **step_params):
    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]
    for i in xrange(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))

    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

        src.reshape(1,3,h,w) # resize the network's input image size
        src.data[0] = octave_base+detail
        for i in xrange(iter_n):
            make_step2(net, end=end, clip=clip, **step_params)

            # visualization
            vis = deprocess(net, src.data[0])
            if not clip: # adjust image contrast if clipping is disabled
                vis = vis*(255.0/np.percentile(vis, 99.98))
            showarray(vis)
            print octave, i, end, vis.shape
            clear_output(wait=True)

        # extract details produced on the current octave
        detail = src.data[0]-octave_base
    # returning the resulting image
    return deprocess(net, src.data[0])
