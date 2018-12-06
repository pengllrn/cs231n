import numpy as np

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W, dtype='f4')

    num_classes = W.shape[1]
    num_train = X.shape[0]

    scores = X.dot(W)
    maxLogC = np.max(scores, axis=1)
    maxLogC = np.reshape(np.repeat(maxLogC, num_classes), scores.shape)
    expScores = np.exp(scores/maxLogC)  #r如何处理，不缩小无法计算，相加减都不行

    for i in range(num_train):
        esum = sum(expScores[i])  # 第i个样本的e和
        eyi = expScores[i, y[i]]
        li = -np.log(eyi / esum)
        loss += li

        for j in range(num_classes):
            dW[:, j] += (expScores[i, j] / esum) * X[i]

        dW[:, y[i]] -= X[i]
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)

    # loss and gradient implement
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    pass
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    num_classes = W.shape[1]
    num_train = W.shape[0]

    scores = X.dot(W)
    maxLogC = np.max(scores, axis=1)
    maxLogC = np.reshape(np.repeat(maxLogC, num_classes), scores.shape)
    expScores = np.exp(scores/maxLogC)  #r如何处理，不缩小无法计算，相加减都不行 N*C

    #loss
    esum=np.sum(expScores,axis=1)  #N*1
    eyi=expScores[np.arange(num_train),y]
    li = -np.log(eyi/esum)
    loss=sum(li)/num_train
    loss += 0.5 * reg * np.sum(W * W)

    #gradient
    expScoresSumRow=np.reshape(np.repeat(np.sum(expScores,axis=1),num_classes),(num_train,num_classes))
    graidentMatrix=expScores/expScoresSumRow #N*C


    #对于yi要-1
    graidentMatrix[np.arange(num_train),y]-=1
    dW=X.T.dot(graidentMatrix)

    dW/=num_train
    dW+=reg*W

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    pass
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

