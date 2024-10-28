import numpy as np

def compute_mse(y, tx, w):
    """Calculate the loss using MSE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    N=y.shape[0]
    L=(1/(2*N))*((y-tx@w)**2)
    return np.sum(L)

def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        grad: a numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    N = y.shape[0]
    error = y - tx.dot(w)
    grad = - tx.T.dot(error) / N
    return grad

def build_poly(x, degree):
    """Function that build a polynomial feature expansion of an array.
    The expansion chosen in that case is:
        given x=np.array([1,2,3],
                         [4,5,6])
        the expansion of degree 2 will be
                np.array([1,1,1,1,2,3,1,4,9],
                         [1,1,1,4,5,6,16,25,36])

    

    Args:
        x (ndarray): the array to be taken in consideration
        degree (int): the degree of the feature expansion

    Returns:
        ndarray: the expanded data 
    """
    poly=np.ones((x.shape[0],x.shape[1]))
    for i in range(1,degree+1):
        expansion=np.power(x,i)
        poly=np.c_[poly,expansion]
    return poly

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
