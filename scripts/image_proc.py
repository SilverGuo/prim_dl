import cv2
import theano
import theano.tensor as T
import numpy as np
from theano.tensor.nnet import conv

__all__ = ['laplacian_pyramid',
           'lecun_lcn',
           'gaussian_filter']

#==============================================================================
# laplacian_pyramid
#==============================================================================
def laplacian_pyramid(img, n=1):
    temp = img
    laplist = []
    for i in range(n):
        down = cv2.pyrDown(temp)
        up = cv2.pyrUp(down)
        laplist.append(temp - up) 
        temp = down
    return laplist


#==============================================================================
# lecun_lcn
#==============================================================================
def lecun_lcn(img, threshold=1e-4, radius=9):
    filter_shape = (1, img.shape[2], radius, radius)
    filters = theano.shared(gaussian_filter(filter_shape), borrow=True)
    
    X = img.transpose(2, 0, 1).reshape(1, img.shape[2], img.shape[0], img.shape[1])
    
    image_shape = X.shape
    
    # Compute the Guassian weighted average by means of convolution
    convout = conv.conv2d(
        input = X,
        filters = filters,
        image_shape = image_shape,
        filter_shape = filter_shape,
        border_mode = 'full'
    )
    
    # Subtractive step
    mid = int(np.floor(filter_shape[2] / 2.))

    # Make filter dimension broadcastable and subtract
    centered_X = X - T.addbroadcast(convout[:, :, mid:-mid, mid:-mid], 1)

    # Compute variances
    sum_sqr_XX = conv.conv2d(
        input = T.sqr(centered_X),
        filters = filters,
        image_shape = image_shape,
        filter_shape = filter_shape,
        border_mode = 'full'
    )


    # Take square root to get local standard deviation
    denom = T.sqrt(sum_sqr_XX[:,:,mid:-mid,mid:-mid])

    per_img_mean = denom.mean(axis=[2,3])
    divisor = T.largest(per_img_mean.dimshuffle(0, 1, 'x', 'x'), denom)
    # Divisise step
    new_X = centered_X / T.maximum(T.addbroadcast(divisor, 1), threshold)
    
    return new_X
    
#==============================================================================
# gaussian_filter
#==============================================================================
def gaussian_filter(kernel_shape):
    x = np.zeros(kernel_shape, dtype=theano.config.floatX)

    def gauss(x, y, sigma=2.0):
        Z = 2 * np.pi * sigma ** 2
        return  1. / Z * np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))

    mid = np.floor(kernel_shape[-1] / 2.)
    for kernel_idx in xrange(0, kernel_shape[1]):
        for i in xrange(0, kernel_shape[2]):
            for j in xrange(0, kernel_shape[3]):
                x[0, kernel_idx, i, j] = gauss(i - mid, j - mid)

    return x / np.sum(x)