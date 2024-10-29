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
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    loss = compute_loss_mse(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_=0.5):
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
    N = tx.shape[0]
    D = tx.shape[1]
    w = np.linalg.solve(tx.T.dot(tx) + lambda_ * 2 * N * np.identity(D), tx.T.dot(y))
    loss = compute_loss_mse(y, tx, w)
    return w, loss


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

############################## WHAT'S NEXT IS BONUS ########################################

def compute_weighted_logistic_loss(y, tx, w, w_vec):
    """
    Compute the weighted logistic loss by negative log likelihood.

    Parameters:
    y : numpy array of shape (N,)
        The target binary labels (0 or 1).
    tx : numpy array of shape (N, D)
        The input data.
    w : numpy array of shape (D,)
        The model weights.
    w_vec : tuple of floats (w0, w1)
        The weights for the positive and negative classes.

    Returns:
    loss : float
        The weighted logistic loss.
    """
    y_pred = sigmoid(np.dot(tx, w))
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -np.mean(
        w_vec[0] * y * np.log(y_pred) + w_vec[1] * (1 - y) * np.log(1 - y_pred)
    )


def compute_weighted_gradient_logistic(y, tx, w, w_vec):
    """
    Compute the gradient of the weighted logistic loss.

    Parameters:
    y : numpy array of shape (N,)
        The target binary labels (0 or 1).
    tx : numpy array of shape (N, D)
        The input data.
    w : numpy array of shape (D,)
        The model weights.
    w_vec : tuple of floats (w0, w1)
        The weights for the positive and negative classes.

    Returns:
    gradient : numpy array of shape (D,)
        The gradient of the weighted logistic loss with respect to w.
    """
    pred = sigmoid(tx.dot(w))
    gradient1 = tx.T.dot(-w_vec[0] * y + w_vec[1] * pred)
    gradient2 = (w_vec[0] - w_vec[1]) * (y * pred).T @ tx
    return (gradient1 + gradient2) / len(y)


def weighted_reg_logistic_regression(
    y, tx, lambda_, initial_w, max_iters, gamma, weight
):
    """
    Perform regularized weighted logistic regression using Gradient Descent.

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
    weight : tuple of floats (w0, w1)
        The weights for the positive and negative classes.

    Returns:
    w : numpy array of shape (D,)
        The optimized weights.
    loss : float
        The weighted logistic loss corresponding to the optimized weights.
    """
    if max_iters == 0:
        return (initial_w, compute_weighted_logistic_loss(y, tx, initial_w, weight))

    w = initial_w
    for iter in range(max_iters):
        gradient = (
            compute_weighted_gradient_logistic(y, tx, w, weight) + 2 * lambda_ * w
        )
        w = w - gamma * gradient
        if iter % 100 == 0:
            loss = compute_weighted_logistic_loss(
                y, tx, w, weight
            ) + lambda_ * np.squeeze(w.T.dot(w))
            print(
                "Current iteration={i}, loss={l} (with regularization)".format(
                    i=iter, l=loss
                )
            )
    loss = compute_weighted_logistic_loss(y, tx, w, weight)
    return w, loss