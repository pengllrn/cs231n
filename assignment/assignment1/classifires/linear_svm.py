import numpy as np

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights. 3072*10
    - X: A numpy array of shape (N, D) containing a minibatch of data. 50000*3072
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)  # 1*10
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, i] += X[i]
                dW[:, y[i]] += (-X[i])

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    dW /= num_train
    dW += reg * W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    num_train = X.shape[0]
    num_classes = W.shape[1]
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    scores = X.dot(W)  # num_train*C
    correct_class_scores = scores[np.arange(num_train), y]  # num_train*1
    # 重复10次，即复制10次正确的score，用于后面矩阵相减 num_train*C
    correct_class_scores = np.reshape(np.repeat(correct_class_scores, num_classes), (num_train, num_classes))
    # 所有分数减去正确分数，并加一,1.0为delta值，表示合页函数的右移
    margin = scores - correct_class_scores + 1.0
    # 为了使最小误差为0，应该使正确的分类的误差矩阵为0
    # num_train * C
    margin[np.arange(num_train), y] = 0

    # margin中所有元素相加，1
    loss = (np.sum(margin[margin > 0])) / num_train
    loss += 0.5 * reg * np.sum(W * W)  # 0.5是为了后面求导方便

    # gradient
    margin[margin > 0] = 1
    margin[margin < 0] = 0

    row_sum = np.sum(margin, axis=1)  # num_train * 1
    margin[np.arange(num_train), y] = -row_sum  # 对真实的那一列的处理
    dW += np.dot(X.T, margin)  # D by C

    dW /= num_train
    dW += reg * W  # 正则函数求导

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW
