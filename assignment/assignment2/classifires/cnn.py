from builtins import object
import numpy as np

from assignment2.cs231n.layers import *
from assignment2.cs231n.fast_layers import *
from assignment2.cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        C, H, W = input_dim
        F = num_filters

        self.params['W1'] = weight_scale * np.random.randn(F, C, filter_size, filter_size)
        self.params['b1'] = np.zeros(F)

        self.params['W2'] = weight_scale * np.random.randn(F * int(H // 2) * (H // 2), hidden_dim)  # 强制让卷积后H和W不变？
        self.params['b2'] = np.zeros(hidden_dim)

        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}  # 由H_=H,反推pad

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        out, conv_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        out, affi1_cache = affine_relu_forward(out, W2, b2)
        out, affi2_cache = affine_forward(out, W3, b3)
        scores = out
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        data_loss, dout = softmax_loss(scores, y)
        W_square_sum = 0
        for layer in range(3):
            Wi = self.params['W%d' % (layer + 1)]
            W_square_sum += np.sum(Wi ** 2)
        reg_loss = 0.5 * self.reg * W_square_sum
        loss = data_loss + reg_loss

        dout, dW3, db3 = affine_backward(dout, affi2_cache)
        dout, dW2, db2 = affine_relu_backward(dout, affi1_cache)
        dout, dW1, db1 = conv_relu_pool_backward(dout, conv_cache)

        dW1 = self.reg * dW1
        dW2 = self.reg * dW2
        dW3 = self.reg * dW3

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        grads['dW3'] = dW3
        grads['db3'] = db3
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def fix(self,X,y,X_val,y_val,lerrn_rate,num_epochs=100,verbose=True):
        for i in range(num_epochs):
            loss,grads=self.loss(X,y)
            for j in range(3):
                self.params['W%d' % (j+1)] += -lerrn_rate * grads['dW%d' % (j+1)]
                self.params['b%d' % (j+1)] += -lerrn_rate * grads['db%d' % (j+1)]

            train_acc = (self.predict(X) == y).mean()
            val_acc = (self.predict(X_val) == y_val).mean()

            if verbose:
                print('Finished epoch %d / %d: loss %f, train_acc: %f, val_acc: %f' % (i, num_epochs, loss, train_acc, val_acc))
    def predict(self,X):
        scores=self.loss(X)
        y_pred = np.argmax(scores, axis=1)
        return y_pred




if __name__ == '__main__':
    # 取数训练20
    a = np.loadtxt('E:/Jupyter/data/tr20.txt', delimiter=',',dtype='i2')

    X = a[:, :3072]
    input=X.reshape(20, 32, 32, 3, order="F")
    input = input.transpose(0, 3, 1, 2)
    y = a[:,3072]
    tlc=ThreeLayerConvNet(reg=1)
    tlc.fix(input,y,input,y,1e-2,100,True)


