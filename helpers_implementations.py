import numpy as np

def compute_gradient_mse(y, tx, w):
    """
    Compute the gradient of the MSE (Mean Squared Error) loss function.

    Parameters:
    y : numpy array of shape (N,)
        The target values.
    tx : numpy array of shape (N, D)
        The input data.
    w : numpy array of shape (D,)
        The model weights.

    Returns:
    gradient : numpy array of shape (D,)
        The gradient of the MSE loss with respect to the weights w.
    """
    N = len(y)
    error = y - tx.dot(w)
    gradient = -1 / N * tx.T.dot(error)
    return gradient

def compute_loss_mse(y, tx, w):
    """
    Compute the Mean Squared Error (MSE) loss.

    Parameters:
    y : numpy array of shape (N,)
        The target values.
    tx : numpy array of shape (N, D)
        The input data.
    w : numpy array of shape (D,)
        The model weights.

    Returns:
    loss : float
        The MSE loss.
    """
    N = len(y)
    squared_error = (y - tx.dot(w)) ** 2
    loss = 1 / (2 * N) * np.sum(squared_error)
    return loss

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.

    Parameters:
    y : numpy array
        Output desired values.
    tx : numpy array
        Input data.
    batch_size : int
        Size of each mini-batch.
    num_batches : int
        Number of batches to generate.
    shuffle : bool
        Whether to shuffle the data before creating batches.

    Yields:
    minibatch_y : numpy array
        Mini-batch of output values.
    minibatch_tx : numpy array
        Mini-batch of input data.
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

#################################### LOGISTIC REGRESSION HELPERS ###############################################

def sigmoid(t):
    """
    Apply the sigmoid function on t.

    Parameters:
    t : numpy array
        Input data.

    Returns:
    s : numpy array
        The sigmoid of the input data.
    """
    t = np.where(t > 500, 500, t)
    t = np.where(t < -500, -500, t)
    return 1.0 / (1.0 + np.exp(-t))


def compute_logistic_loss(y, tx, w):
    """
    Compute the cost by negative log likelihood for logistic regression.

    Parameters:
    y : numpy array of shape (N,)
        The target binary labels (0 or 1).
    tx : numpy array of shape (N, D)
        The input data.
    w : numpy array of shape (D,)
        The model weights.

    Returns:
    loss : float
        The negative log likelihood loss.
    """
    y_pred = sigmoid(np.dot(tx, w))
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))


def compute_gradient_logistic(y, tx, w):
    """
    Compute the gradient of the logistic loss.

    Parameters:
    y : numpy array of shape (N,)
        The target binary labels (0 or 1).
    tx : numpy array of shape (N, D)
        The input data.
    w : numpy array of shape (D,)
        The model weights.

    Returns:
    gradient : numpy array of shape (D,)
        The gradient of the logistic loss with respect to w.
    """
    pred = sigmoid(tx.dot(w))
    gradient = tx.T.dot(pred - y)
    return gradient / len(y)