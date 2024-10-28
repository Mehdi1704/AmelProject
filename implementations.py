import numpy as np
import matplotlib.pyplot as plt
from helpers import *
from helpers_logistic import *
from helpers_implementations import *

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: numpy array of shape (D, ). The final model parameters after GD
        loss: The corresponding loss value (cost function) for the final model parameters
    """
    w = initial_w
    N = y.shape[0]
    for _ in range(max_iters):
        error = y - tx.dot(w)
        gradient = - tx.T.dot(error) / N
        w = w - gamma * gradient
        
    loss = compute_mse(y, tx, w)
    return w, loss 

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Stochastic Gradient Descent (SGD) algorithm with mini-batch size 1.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        w: numpy array of shape (2, ). The final model parameters after SGD
        loss: The corresponding loss value (cost function) for the final model parameters
    """
    w = initial_w
    for _ in range(max_iters):
        # Randomly shuffle the data
        indices = np.random.permutation(np.arange(len(y)))
        for y_batch, tx_batch in batch_iter(
            y, tx, batch_size=1, num_batches=1
        ):
            gradient = compute_gradient(y_batch, tx_batch, w)
            w = w - gamma * gradient
        
    loss = compute_mse(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    N=y.shape[0]
    D=tx.shape[1]
    w=np.zeros(D)

    A=tx.T@tx ## DxD
    b=tx.T@y ## D
    w=np.linalg.lstsq(A,b)[0]
    loss=compute_mse(y,tx,w)
    
    return w , loss


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    D=tx.shape[1]
    N=tx.shape[0]
    A=tx.T@tx+(2*N*lambda_)*np.identity(D)
    b=tx.T@y
    w=np.linalg.solve(A,b)
    loss=compute_mse(y,tx,w)
    return w,loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Implement Logistic Regression by Gradient Descent

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        initial_w : numpy array of shape (D,) with the inital model parameters
        max_iters (_type_): the maximum number of iteration
        gamma (_type_): Gradient Descent Stepsize

    Returns:
        w: optimized parameters
        loss: logistic loss of the model
    """
    threshold=1e-8
    losses=[]
    w=initial_w
    for iter in range(max_iters):
        loss, w=learning_by_gradient_descent(y,tx,w,gamma)
        losses.append(loss)
        if len(losses)>1 and np.abs(losses[-1]-losses[-2])<threshold:
            break
    return w , loss


def reg_logistic_regression(y, tx, lambda_ ,initial_w, max_iters, gamma):
    """Implement Penalized Logistic Regression by Gradient Descent

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_ : penalisation scalar
        initial_w : numpy array of shape (D,) with the inital model parameters
        max_iters (_type_): the maximum number of iteration
        gamma (_type_): Gradient Descent Stepsize

    Returns:
        w: optimized parameters
        loss: logistic loss of the model
    """    
    w=initial_w
    threshold=1e-8
    losses=[]
    for iter in range(max_iters):
        norm_w=np.linalg.norm(w)
        loss=calculate_loss(y,tx,w)
        gradient=calculate_gradient(y,tx,w)
        loss+=lambda_*norm_w*norm_w
        gradient+=2*lambda_*w
        w-=gamma*gradient
        losses.append(loss)
        if len(losses)>1 and np.abs(losses[-1]-losses[-2])<threshold:
            break
    return w, loss