import cv2
import theano
import theano.tensor as T
import numpy as np
from theano.tensor.nnet import conv
from PIL import Image

__all__ = ['open_convert',
           'show_image',
           'gaussian_pyramid',
           'laplacian_pyramid',
           'lecun_lcn',
           'gaussian_filter']

#==============================================================================
# open_convert
#==============================================================================
def open_convert(img_file, mode):
    img = Image.open(img_file)
    img = np.asarray(img.convert(mode))
    return img

#==============================================================================
# show_image
#==============================================================================
def show_image(img, mode=None):
    img = Image.fromarray(img, mode=mode)
    img.show()
    return

#==============================================================================
# gaussian_pyramid
#==============================================================================
def gaussian_pyramid(img, n=1):
    gp = img.copy()
    gplist = [gp]
    for i in xrange(n-1):
        gp = cv2.pyrDown(gp)
        gplist.append(gp)
    return gplist

#==============================================================================
# laplacian_pyramid
#==============================================================================
def laplacian_pyramid(img, n=1):
    gplist = gaussian_pyramid(img, n=n+1)
    lplist = []
    for i in xrange(n):
        ge = cv2.pyrUp(gplist[i+1], dstsize=(gplist[i].shape[1], gplist[i].shape[0]))
        lp = cv2.subtract(gplist[i], ge)
        lplist.append(lp)
    return lplist

#==============================================================================
# lecun_lcn
#==============================================================================
def lecun_lcn(input, kernel_shape=9, threshold=1e-4):
    input = np.float64(input)
    
    X = input.transpose(2, 0, 1).reshape(input.shape[2], 1, input.shape[0], input.shape[1])
    
    filter_shape = (1, 1, kernel_shape, kernel_shape)
    filters = gaussian_filter(kernel_shape).reshape(filter_shape)
    filters = theano.shared(theano._asarray(filters, dtype=theano.config.floatX), borrow=True)

    convout = conv.conv2d(input=X,
                          filters=filters,
                          filter_shape=filter_shape,
                          border_mode='full')

    mid = int(np.floor(kernel_shape / 2.))
    centered_X = X - convout[:, :, mid:-mid, mid:-mid]

    sum_sqr_XX = conv.conv2d(input=centered_X ** 2,
                             filters=filters,
                             filter_shape=filter_shape,
                             border_mode='full')

    denom = T.sqrt(sum_sqr_XX[:, :, mid:-mid, mid:-mid])
    per_img_mean = denom.mean(axis=[2, 3])
    divisor = T.largest(per_img_mean.dimshuffle(0, 1, 'x', 'x'), denom)
    
    divisor = T.maximum(divisor, threshold)

    new_X = centered_X / divisor

    output = new_X.eval().reshape(input.shape[2], input.shape[0], input.shape[1]).transpose(1, 2 ,0)
    return output

#==============================================================================
# gaussian_filter
#==============================================================================
def gaussian_filter(kernel_shape):
    x = np.zeros((kernel_shape, kernel_shape),
                    dtype=theano.config.floatX)

    def gauss(x, y, sigma=2.0):
        Z = 2 * np.pi * sigma ** 2
        return 1. / Z * np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))

    mid = np.floor(kernel_shape / 2.)
    for i in xrange(0, kernel_shape):
        for j in xrange(0, kernel_shape):
            x[i, j] = gauss(i - mid, j - mid)

    return x / np.sum(x)
