import caffe
import numpy as np

class TransposeLayer(caffe.Layer):

    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        #top[0].reshape(*bottom[0].data.shape)
        if(len(bottom)>1):
            t_shape = list(bottom[1].data.shape)
            t_shape[1] = bottom[0].channels
            t_shape = tuple(t_shape)
            top[0].reshape(*t_shape)
        else:
            t_shape = (bottom[0].num*bottom[0].height*bottom[0].width,bottom[0].channels)
            top[0].reshape(*t_shape)

    def forward(self, bottom, top):
        #top[0].data[...] = self.forward_theano(bottom[0].data[...])
        if len(bottom)>1:
            tmp_data = bottom[0].data.reshape((top[0].height,top[0].width,top[0].num,top[0].channels)).copy()
            top[0].data[...] = np.transpose(tmp_data,(2,3,0,1)).copy()[...]
        else:
            top[0].data[...] = np.transpose(bottom[0].data,(2,3,0,1)).reshape(top[0].data.shape).copy()[...]

    def backward(self, top, prop_down, bottom):
        if len(bottom)>1:
            bottom[0].diff[...] =  np.transpose(top[0].diff,(2,3,0,1)).reshape(bottom[0].diff.shape)[...]
        else:
            tmp_diff = top[0].diff.reshape((bottom[0].height,bottom[0].width,bottom[0].num,bottom[0].channels)).copy()
            bottom[0].diff[...] = np.transpose(tmp_diff,(2,3,0,1)).copy()[...]
