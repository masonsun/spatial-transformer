import numpy as np
import theano
import theano.tensor as T

from theano.tensor.nnet import conv
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
from nn_utils import *

class SpatialTransformerLayer(object):
    def __init__(self, conv_input, theta, down_fraction=1):    
        self.down_fraction = down_fraction
        self.output = affine_sampling(theta, conv_input, self.down_fraction)

def affine_sampling(theta, input, df):
    num_batch, num_channels, height, width = input.shape
    theta = T.reshape(theta, (-1, 2, 3))
    f_height = T.cast(height, 'float32')
    f_width = T.cast(width, 'float32')
    o_height = T.cast(f_height // df, 'int64')
    o_width = T.cast(f_width // df, 'int64')

    grid = create_meshgrid(o_height, o_width)

    # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
    Tg = T.dot(theta, grid)
    xs, ys = Tg[:, 0], Tg[:, 1]
    xs_flat = xs.flatten()
    ys_flat = ys.flatten()

    # dimshuffle input to (bs, height, width, channels)
    input_dim = input.dimshuffle(0, 2, 3, 1)
  
    input_trans = bilinear_sampling(
        input_dim, xs_flat, ys_flat,
        df)
  
    output = T.reshape(input_trans,
                       (num_batch, o_height, o_width, num_channels))
    output = output.dimshuffle(0, 3, 1, 2) 
    return output

def create_linspace(start, stop, num):
    start = T.cast(start, 'float32')
    stop = T.cast(stop, 'float32')
    num = T.cast(num, 'float32')
    step = (stop-start)/(num-1)
    return T.arange(num, dtype='float32')*step+start

def create_meshgrid(height, width):
    xt = T.dot(T.ones((height, 1)),
               create_linspace(-1.0, 1.0, width).dimshuffle('x', 0))
    yt = T.dot(create_linspace(-1.0, 1.0, height).dimshuffle(0, 'x'),
                 T.ones((1, width)))
   
    xt_flat = xt.reshape((1, -1))
    yt_flat = yt.reshape((1, -1))
    ones = T.ones_like(xt_flat)
    grid = T.concatenate([xt_flat, yt_flat, ones], axis=0)
    return grid

def rept(x, n_rep):
    rep = T.ones((n_rep,), dtype='int32').dimshuffle('x', 0)
    x = T.dot(x.reshape((-1, 1)), rep)
    return x.flatten()

def binlinear_sampling(img, x, y, df):
    # constants
    num_batch, height, width, channels = img.shape
    f_height = T.cast(height, 'float32')
    f_width = T.cast(width, 'float32')
    o_height = T.cast(f_height // downsample_factor, 'int64')
    o_width = T.cast(f_width // downsample_factor, 'int64')
    zero = T.zeros([], dtype='int64')
    y_max = T.cast(img.shape[1] - 1, 'int64')
    x_max = T.cast(img.shape[2] - 1, 'int64')
    o_x = (x + 1.0)*(f_width) / 2.0
    o_y = (y + 1.0)*(f_height) / 2.0
  
    x0 = T.cast(T.floor(o_x), 'int64')
    x1 = x0 + 1
    y0 = T.cast(T.floor(o_y), 'int64')
    y1 = y0 + 1

    x_floor = T.clip(x0, zero, x_max)
    x_ceil = T.clip(x1, zero, x_max)
    y_floor = T.clip(y0, zero, y_max)
    y_ceil = T.clip(y1, zero, y_max)
    dim1 = width*height
    dim2 = width
    base = rept(
        T.arange(num_batch, dtype='int32')*dim1, o_height*o_width)
    base_y_floor = base + y_floor*dim2
    base_y_ceil = base + y_ceil*dim2
    idxa = base_y_floor + x_floor
    idxb = base_y_ceil + x_floor
    idxc = base_y_floor + x_ceil
    idxd = base_y_ceil + x_ceil


    img_flat = img.reshape((-1, channels))
    I_a = img_flat[idxa]
    I_b = img_flat[idxb]
    I_c = img_flat[idxc]
    I_d = img_flat[idxd]

    # and finanly calculate interpolated values
    xf_f = T.cast(x_floor, 'float32')
    xc_f = T.cast(x_ceil, 'float32')
    yf_f = T.cast(y_floor, 'float32')
    yc_f = T.cast(y_ceil, 'float32')
    w_a = ((xc_f-x) * (yc_f-y)).dimshuffle(0, 'x')
    w_b = ((xc_f-x) * (y-yf_f)).dimshuffle(0, 'x')
    w_c = ((x-xf_f) * (yc_f-y)).dimshuffle(0, 'x')
    w_d = ((x-xf_f) * (y-yf_f)).dimshuffle(0, 'x')
    output = T.sum([w_a*I_a, w_b*I_b, w_c*I_c, w_d*I_d], axis=0)
    return output

class STN_CNN(object):
    def __init__(self, input_dim, img, nconvs = [20,20], downsampling = 1, scale = 1):
        input = img
        scale = int(scale)

        if scale != 1:
            input = T.signal.pool.pool_2d(img, ds = (scale,scale), ignore_border = True)

        rng = np.random.RandomState(12345)
        batch_size, channels, height, width = input_dim
        height /= scale
        width /= scale

        layer0 = ConvPoolLayer(
                rng,
                input=input,
                image_shape=(batch_size, channels, height, width),
                filter_shape=(nconvs[0], channels, 5, 5),
                poolsize=(2, 2)
        )

        layer1 = ConvPoolLayer(
                rng,
                input=layer0.output,
                image_shape=(batch_size, nconvs[0], (height - 4) / 2, (width - 4) / 2),
                filter_shape=(nconvs[1], nconvs[0], 5, 5),
                poolsize=(1, 1)
        )
        
        layer2_input = layer1.output.flatten(2)

        # construct a fully-connected sigmoidal layer
        layer2 = HiddenLayer(
            rng,
            input=layer2_input,
            n_in= nconvs[1] * ((height - 4) / 2 - 4) * ((width - 4) / 2 - 4),
            n_out=20,
            activation = T.nnet.relu
        )

        print("...initialize localization to identity")

        W_values = np.zeros((20, 6), dtype = theano.config.floatX)
        W = theano.shared(value=W_values, name='W', borrow=True)
        b_values = np.array([1,0,0,0,1,0],dtype = theano.config.floatX)
        b = theano.shared(value=b_values, name='b', borrow=True)

        layer3 = HiddenLayer(
            rng,
            input=layer2.output,
            n_in= 20,
            n_out=6,
            activation = None,
            W = W,
            b = b
        )

        theta = layer3.output
        theta = T.reshape(theta, (-1,6))
        self.input = input
        self.params = layer0.params + layer1.params + layer2.params + layer3.params
        self.output = SpatialTransformerLayer(input, theta, down_fraction = downsampling).output
        self.theta = theta

class STN_FCN(object):
    def __init__(self, input_dim, input, nhids = [32,32,32], downsampling = 1, activation = T.nnet.relu):
        batch_size, channel, height, width = input_dim
        img_size = height * width
        inputX = input.reshape((batch_size, channel, img_size))
        rng = np.random.RandomState(12345)

        layer0 = HiddenLayer(
            rng,
            input=inputX,
            n_in= img_size,
            n_out=nhids[0],
            activation = activation
        )

        layer1 = HiddenLayer(
            rng,
            input=layer0.output,
            n_in= nhids[0],
            n_out= nhids[1],
            activation =  activation
        )

        layer2 = HiddenLayer(
            rng,
            input=layer1.output,
            n_in= nhids[1],
            n_out= nhids[2],
            activation =  activation
        )
        
        print("...initialize localization to identity")
        
        W_values = np.zeros((nhids[1], 6), dtype = theano.config.floatX)
        W = theano.shared(value=W_values, name='W', borrow=True)
        b_values = np.array([1,0,0,0,1,0],dtype = theano.config.floatX)
        b = theano.shared(value=b_values, name='b', borrow=True)
        
        layer3 = HiddenLayer(
            rng,
            input=layer2.output,
            n_in= nhids[2],
            n_out= 6,
            activation = None,
            W = W, 
            b = b
        )
        
        theta = layer3.output
        theta = T.reshape(theta, (-1,6))
        self.params = layer0.params + layer1.params + layer2.params + layer3.params
        self.output = SpatialTransformerLayer(input, theta, down_fraction = downsampling).output
        self.theta = theta
        self.input = input
