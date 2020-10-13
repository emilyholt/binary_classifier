import os
import numpy as np
import pandas as pd

import sklearn.linear_model
import sklearn.metrics
import matplotlib
import matplotlib.pyplot as plt

from helpers import calc_mean_binary_cross_entropy_from_probas

# Data loading and globals
DATA_DIR = os.path.join('data_digits_8_vs_9_noisy')  # Make sure you have downloaded data and your directory is correct

x_tr_M784 = np.loadtxt(os.path.join(DATA_DIR, 'x_train.csv'), delimiter=',', skiprows=1)
x_va_N784 = np.loadtxt(os.path.join(DATA_DIR, 'x_valid.csv'), delimiter=',', skiprows=1)
x_te_N784 = np.loadtxt(os.path.join(DATA_DIR, 'x_test.csv'), delimiter=',', skiprows=1)

M_shape = x_tr_M784.shape
N_shape = x_va_N784.shape

N = N_shape[0]
M = M_shape[0]

y_tr_M = np.loadtxt(os.path.join(DATA_DIR, 'y_train.csv'), delimiter=',', skiprows=1)
y_va_N = np.loadtxt(os.path.join(DATA_DIR, 'y_valid.csv'), delimiter=',', skiprows=1)

# Functions


def datasetExploration():
    tr_total = M
    va_total = N

    tr_pos_label = np.count_nonzero(y_tr_M)
    va_pos_label = np.count_nonzero(y_va_N)

    tr_frac = np.around(tr_pos_label / tr_total, 3)
    va_frac = np.around(va_pos_label / va_total, 3)

    print(f"""
    Number of examples, total     | {tr_total}  | {va_total}
    Number of positive examples   | {tr_pos_label}  | {va_pos_label}
    Fraction of positive examples | {tr_frac:.3f} | {va_frac:.3f}
    """)


def lossAssessment():
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
    plotLossError(iterations, iter_step, error_tr_lr_i, error_va_lr_i, bce_tr_lr_i, bce_va_lr_i,)


def plotLossError(iterations, iter_step, error_tr_lr_i, error_va_lr_i, bce_tr_lr_i, bce_va_lr_i,):
    _, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=False)

    # Set up the Log Loss subplot
    axes[0].plot(np.arange(1, iterations + 1, iter_step), bce_tr_lr_i, 'b.-', label='train')
    axes[0].plot(np.arange(1, iterations + 1, iter_step), bce_va_lr_i, 'r.-', label='valid')
    axes[0].set_title('Log Loss')
    axes[0].set_ylabel('loss')
    axes[0].set_xlabel("number of iterations")
    axes[0].legend()

    # Set up the Error rate subplot
    axes[1].plot(np.arange(1, iterations + 1, iter_step), error_tr_lr_i, 'b:', label='train')
    axes[1].plot(np.arange(1, iterations + 1, iter_step), error_va_lr_i, 'r:', label='valid')
    axes[1].set_title('Error Rate')
    axes[1].set_ylabel('error')
    axes[1].set_xlabel("number of iterations")
    axes[1].legend()

    plt.show()


if __name__ == "__main__":
    # datasetExploration()
    lossAssessment()
