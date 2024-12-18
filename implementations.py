import numpy as np
from helpers_implementations import *


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Perform Gradient Descent to minimize MSE loss.

    Parameters:
    y : numpy array of shape (N,)
        The target values.
    tx : numpy array of shape (N, D)
        The input data.
    initial_w : numpy array of shape (D,)
        The initial weights.
    max_iters : int
        The maximum number of iterations.
    gamma : float
        The learning rate.

    Returns:
    w : numpy array of shape (D,)
        The optimized weights.
    loss : float
        The MSE loss corresponding to the optimized weights.
    """
    if max_iters == 0:
        return (initial_w, compute_loss_mse(y, tx, initial_w))
    w = initial_w
    for iter in range(max_iters):
        gradient = compute_gradient_mse(y, tx, w)
        w = w - gamma * gradient
        if iter % 100 == 0:
            print(
                "Current iteration={i}, loss={l}".format(
                    i=iter, l=compute_loss_mse(y, tx, w)
                )
            )
    loss = compute_loss_mse(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Perform Stochastic Gradient Descent to minimize MSE loss.

    Parameters:
    y : numpy array of shape (N,)
        The target values.
    tx : numpy array of shape (N, D)
        The input data.
    initial_w : numpy array of shape (D,)
        The initial weights.
    max_iters : int
        The maximum number of iterations.
    gamma : float
        The learning rate.

    Returns:
    w : numpy array of shape (D,)
        The optimized weights.
    loss : float
        The MSE loss corresponding to the optimized weights.
    """
    if max_iters == 0:
        return (initial_w, compute_loss_mse(y, tx, initial_w))
    w = initial_w
    for iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):
            gradient = compute_gradient_mse(minibatch_y, minibatch_tx, w)
            w = w - gamma * gradient
        if iter % 100 == 0:
            print(
                "Current iteration={i}, loss={l}".format(
                    i=iter, l=compute_loss_mse(y, tx, w)
                )
            )

    loss = compute_loss_mse(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """
    Calculate the least squares solution using the normal equations.

    Parameters:
    y : numpy array of shape (N,)
        The target values.
    tx : numpy array of shape (N, D)
        The input data.

    Returns:
    w : numpy array of shape (D,)
        The optimal weights minimizing the MSE.
    loss : float
        The MSE loss corresponding to the optimal weights.
    """
    N=y.shape[0]
    D=tx.shape[1]
    w=np.zeros(D)

    A=tx.T@tx
    b=tx.T@y
    w=np.linalg.lstsq(A,b)[0]
    loss=compute_loss_mse(y,tx,w)
    
    return w , loss


def ridge_regression(y, tx, lambda_):
    """
    Implement Ridge Regression using the normal equations.

    Parameters:
    y : numpy array of shape (N,)
        The target values.
    tx : numpy array of shape (N, D)
        The input data.
    lambda_ : float
        Regularization parameter.

    Returns:
    w : numpy array of shape (D,)
        The optimal weights minimizing the regularized MSE.
    loss : float
        The MSE loss corresponding to the optimal weights.
    """
    D=tx.shape[1]
    N=tx.shape[0]
    A=tx.T@tx+(2*N*lambda_)*np.identity(D)
    b=tx.T@y
    w=np.linalg.solve(A,b)
    loss=compute_loss_mse(y,tx,w)
    return w,loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Perform logistic regression using Gradient Descent.

    Parameters:
    y : numpy array of shape (N,)
        The target binary labels (0 or 1).
    tx : numpy array of shape (N, D)
        The input data.
    initial_w : numpy array of shape (D,)
        The initial weights.
    max_iters : int
        The maximum number of iterations.
    gamma : float
        The learning rate.

    Returns:
    w : numpy array of shape (D,)
        The optimized weights.
    loss : float
        The logistic loss corresponding to the optimized weights.
    """
    if max_iters == 0:
        return (initial_w, compute_logistic_loss(y, tx, initial_w))

    w = initial_w
    for iter in range(max_iters):
        gradient = compute_gradient_logistic(y, tx, w)
        w = w - gamma * gradient
        if iter % 100 == 0:
            print(
                "Current iteration={i}, loss={l}".format(
                    i=iter, l=compute_logistic_loss(y, tx, w)
                )
            )
    loss = compute_logistic_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Perform regularized logistic regression using Gradient Descent.

    Parameters:
    y : numpy array of shape (N,)
        The target binary labels (0 or 1).
    tx : numpy array of shape (N, D)
        The input data.
    lambda_ : float
        Regularization parameter.
    initial_w : numpy array of shape (D,)
        The initial weights.
    max_iters : int
        The maximum number of iterations.
    gamma : float
        The learning rate.

    Returns:
    w : numpy array of shape (D,)
        The optimized weights.
    loss : float
        The logistic loss corresponding to the optimized weights.
    """
    if max_iters == 0:
        return (initial_w, compute_logistic_loss(y, tx, initial_w))

    w = initial_w
    for iter in range(max_iters):
        gradient = compute_gradient_logistic(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient
        if iter % 100 == 0:
            loss = compute_logistic_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
            print(
                "Current iteration={i}, loss={l} (with regularization)".format(
                    i=iter, l=loss
                )
            )
    loss = compute_logistic_loss(y, tx, w)
    return w, loss