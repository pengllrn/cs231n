from builtins import range
import numpy as np


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    """
    :param x_shape:
    :param field_height:
    :param field_width:
    :param padding:
    :param stride:
    :return:
    tuple :
    k:kernel,卷积核的核，shape（field_height*field_width*Channel，1）,Channel为通道数，图像有RGB三通道
    i:
    j:
    """
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = (H + 2 * padding - field_height) // stride + 1 #x_pad  H_
    out_width = (W + 2 * padding - field_width) // stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)  #[0,0,..,0,1,...,1,...,field_width-1,...,field_width-1]  shape(field_height*field_width,)
    i0 = np.tile(i0, C)  # shape(field_height*field_width*C,)
    i1 = stride * np.repeat(np.arange(out_height), out_width)  #shape(H_*W_,)
    j0 = np.tile(np.arange(field_width), field_height * C) #shape(field_height*field_width*C,)
    j1 = stride * np.tile(np.arange(out_width), out_height) #shape(W_*H_,)
    #i: shape(field_height*field_width*C,1) + shape(1,W_*H_)=shape(field_height*field_width*C,W_*H_)
    #j:shape(field_height*field_width*C,W_*H_)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1) #shape(field_height*field_width*C，1)

    return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                 stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                                 stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]

pass

if __name__ == '__main__':
    a=np.array(np.random.rand(2,3,3,3)*255,dtype='i2')
    print(a)
    b=im2col_indices(a,3,3)
    c=col2im_indices(b,(2,3,3,3))
    print("----------------")
    print(c)
