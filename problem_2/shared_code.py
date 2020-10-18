# Code stolen from dtphelan1
# Reused for problem 2

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sklearn.linear_model
import sklearn.metrics

def calc_mean_binary_cross_entropy_from_probas(ytrue_N, yproba1_N):
    return sklearn.metrics.log_loss(ytrue_N, yproba1_N, labels=[0, 1]) / np.log(2.0)

def lossAssessment(x_tr_M784, y_tr_M, x_va_N784, y_va_N):
    # Using sklearn.linear_model.LogisticRegression, you should fit a logistic regression models to your training split.
    C = 1e6
    solver = 'lbfgs'
    iterations = 40
    iter_step = 1

    lr_models_i = list()

    error_tr_lr_i = list()
    error_va_lr_i = list()

    bce_tr_lr_i = list()
    bce_va_lr_i = list()

    for i in range(0, iterations, iter_step):
        lr_i = sklearn.linear_model.LogisticRegression(max_iter=i+1, C=C, solver=solver)
        lr_i.fit(x_tr_M784, y_tr_M)  # Part b
        lr_models_i.append(lr_i)

        yproba1_tr_M = lr_i.predict_proba(x_tr_M784)[:, 1]  # The probability of predicting class 1 on the training set
        yproba1_va_N = lr_i.predict_proba(x_va_N784)[:, 1]  # The probability of predicting class 1 on the validation set

        error_tr_lr_i.append(sklearn.metrics.zero_one_loss(y_tr_M, yproba1_tr_M >= 0.5))
        error_va_lr_i.append(sklearn.metrics.zero_one_loss(y_va_N, yproba1_va_N >= 0.5))

        bce_tr_lr_i.append(calc_mean_binary_cross_entropy_from_probas(y_tr_M, yproba1_tr_M))
        bce_va_lr_i.append(calc_mean_binary_cross_entropy_from_probas(y_va_N, yproba1_va_N))
    
    # Plot error & losses
    plot_loss_over_iters(iterations, iter_step, bce_tr_lr_i, bce_va_lr_i,)
    plot_error_over_iters(iterations, iter_step, error_tr_lr_i, error_va_lr_i)

    # Find C value with best bce & error rates
    best_error_ind = error_va_lr_i.index(min(error_va_lr_i))
    best_bce_ind = bce_va_lr_i.index(min(bce_va_lr_i))
    print(f"number of iterations with lowest error = {best_error_ind + 1}")
    print(f"number of iterations with lowest bce = {best_bce_ind + 1}")

def c_selection(x_tr_M784, y_tr_M, x_va_N784, y_va_N):
    # Using sklearn.linear_model.LogisticRegression, you should fit a logistic regression models to your training split.
    C_grid = np.logspace(-9, 6, 31)
    solver = 'lbfgs'
    max_iter = 1000

    lr_models_i = list()
    error_tr_lr_i = list()
    error_va_lr_i = list()
    bce_tr_lr_i = list()
    bce_va_lr_i = list()

    for C in C_grid:
        lr_i = sklearn.linear_model.LogisticRegression(max_iter=max_iter, C=C, solver=solver)
        lr_i.fit(x_tr_M784, y_tr_M)
        lr_models_i.append(lr_i)

        yproba1_tr_M = lr_i.predict_proba(x_tr_M784)[:, 1]  # The probability of predicting class 1 on the training set
        yproba1_va_N = lr_i.predict_proba(x_va_N784)[:, 1]  # The probability of predicting class 1 on the validation set

        error_tr_lr_i.append(sklearn.metrics.zero_one_loss(y_tr_M, yproba1_tr_M >= 0.5))
        error_va_lr_i.append(sklearn.metrics.zero_one_loss(y_va_N, yproba1_va_N >= 0.5))

        bce_tr_lr_i.append(calc_mean_binary_cross_entropy_from_probas(y_tr_M, yproba1_tr_M))
        bce_va_lr_i.append(calc_mean_binary_cross_entropy_from_probas(y_va_N, yproba1_va_N))
    
    # Plot error & losses
    plot_error_over_c(C_grid, error_tr_lr_i, error_va_lr_i)
    plot_loss_over_c(C_grid, bce_tr_lr_i, bce_va_lr_i)

    # Find C value with best bce & error rates
    best_error_ind = error_va_lr_i.index(min(error_va_lr_i))
    best_bce_ind = bce_va_lr_i.index(min(bce_va_lr_i))
    print(f"C-value with lowest error = {C_grid[best_error_ind]}")
    print(f"C-value with lowest bce = {C_grid[best_bce_ind]}")


def plot_loss_over_iters(iterations, iter_step, bce_tr_lr_i, bce_va_lr_i):
    fig, ax = plt.subplots()

    # Set up the Log Loss subplot
    ax.plot(np.arange(1, iterations + 1, iter_step), bce_tr_lr_i, 'b.-', label='train')
    ax.plot(np.arange(1, iterations + 1, iter_step), bce_va_lr_i, 'r.-', label='valid')
    ax.set_title('Log Loss over iterations')
    ax.set_ylabel('loss')
    ax.set_xlabel("number of iterations")
    ax.legend()
    plt.savefig('LogLossIterations.png')
    plt.show()

def plot_error_over_iters(iterations, iter_step, error_tr_lr_i, error_va_lr_i):
    fig, ax = plt.subplots()

    # Set up the Error rate subplot
    ax.plot(np.arange(1, iterations + 1, iter_step), error_tr_lr_i, 'b:', label='train')
    ax.plot(np.arange(1, iterations + 1, iter_step), error_va_lr_i, 'r:', label='valid')
    ax.set_title('Error Rate over iterations')
    ax.set_ylabel('error')
    ax.set_xlabel("number of iterations")
    ax.legend()
    plt.savefig('ErrorRateIterations.png')
    plt.show()

def plot_loss_over_c(C_grid, bce_tr_lr_i, bce_va_lr_i):
    fig, ax = plt.subplots()

    # Set up the Log Loss subplot
    x = np.log10(C_grid)
    ax.plot(x, bce_tr_lr_i, 'b.-', label='train')
    ax.plot(x, bce_va_lr_i, 'r.-', label='valid')
    ax.set_title('Log Loss at different C Values')
    ax.set_ylabel('loss')
    ax.set_xlabel("log_{10} C values")
    ax.legend()
    plt.savefig('LogLossCValues.png')
    plt.show()

def plot_error_over_c(C_grid, error_tr_lr_i, error_va_lr_i):
    fig, ax = plt.subplots()
    
    # Set up the Error rate subplot
    x = np.log10(C_grid)
    ax.plot(x, error_tr_lr_i, 'b:', label='train')
    ax.plot(x, error_va_lr_i, 'r:', label='valid')
    ax.set_title('Error Rate at different C Values')
    ax.set_ylabel('Error Rate')
    ax.set_xlabel("log_{10} C values")
    ax.legend()
    plt.savefig('ErrorRateCValues.png')
    plt.show()

    