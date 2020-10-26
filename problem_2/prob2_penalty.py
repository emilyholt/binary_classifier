import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

import sklearn.linear_model
import sklearn.metrics
import sklearn.neighbors
import sklearn.pipeline

from shared_code import calc_mean_binary_cross_entropy_from_probas

RANDOM_STATE = 1234

DATA_DIR = 'data_sneaker_vs_sandal'
x_tr_M784 = np.loadtxt(os.path.join(DATA_DIR, 'x_train_set.csv'), delimiter=',')
x_va_N784 = np.loadtxt(os.path.join(DATA_DIR, 'x_valid_set.csv'), delimiter=',')

y_tr_M = np.loadtxt(os.path.join(DATA_DIR, 'y_train_set.csv'), delimiter=',')
y_va_N = np.loadtxt(os.path.join(DATA_DIR, 'y_valid_set.csv'), delimiter=',')


def penaltySelection(pickle_it=True):
    """ Iterate over our C_grid to see which hyperparameter gives us the best performance
    """
    # C_grid = np.logspace(-9, 6, 31)
    C = 0.31622776601683794
    solver = 'lbfgs'
    max_iter = 1000  # Fixed per instructions

    # No penalty
    # Build and evaluate model for this C value
    lr_no_penalty = sklearn.linear_model.LogisticRegression(penalty='none', max_iter=max_iter, C=C, solver=solver, random_state=RANDOM_STATE)
    lr_no_penalty.fit(x_tr_M784, y_tr_M)  # Part b

    # Record training and validation set error rate
    yproba1_tr_M_no_penalty = lr_no_penalty.predict_proba(x_tr_M784)[:, 1]  # The probability of predicting class 1 on the training set
    yproba1_va_N_no_penalty = lr_no_penalty.predict_proba(x_va_N784)[:, 1]  # The probability of predicting class 1 on the validation set

    error_train_no_penalty = sklearn.metrics.zero_one_loss(y_tr_M, yproba1_tr_M_no_penalty >= 0.5)
    error_valid_no_penalty = sklearn.metrics.zero_one_loss(y_va_N, yproba1_va_N_no_penalty >= 0.5)
    print(f"No Penalty\n  Train Error = {error_train_no_penalty}; Valid Error = {error_valid_no_penalty}")

    bce_train_no_penalty = calc_mean_binary_cross_entropy_from_probas(y_tr_M, yproba1_tr_M_no_penalty)
    bce_valid_no_penalty = calc_mean_binary_cross_entropy_from_probas(y_va_N, yproba1_va_N_no_penalty)
    print(f"No Penalty\n  Train BCE = {bce_train_no_penalty}; Valid BCE = {bce_valid_no_penalty}")

    ########################################################

    # L1 Penalty
    # Build and evaluate model for this C value
    # NOTE: Changing the solver here because LBGFS isn't compatible with L1 loss penalty
    lr_l1 = sklearn.linear_model.LogisticRegression(penalty='l1', max_iter=max_iter, C=C, solver='liblinear', random_state=RANDOM_STATE)
    lr_l1.fit(x_tr_M784, y_tr_M)  # Part b

    # Record training and validation set error rate
    yproba1_tr_M_l1 = lr_l1.predict_proba(x_tr_M784)[:, 1]  # The probability of predicting class 1 on the training set
    yproba1_va_N_l1 = lr_l1.predict_proba(x_va_N784)[:, 1]  # The probability of predicting class 1 on the validation set

    error_train_l1 = sklearn.metrics.zero_one_loss(y_tr_M, yproba1_tr_M_l1 >= 0.5)
    error_valid_l1 = sklearn.metrics.zero_one_loss(y_va_N, yproba1_va_N_l1 >= 0.5)
    print(f"L1 Penalty\n  Train Error = {error_train_l1}; Valid Error = {error_valid_l1}")

    bce_train_l1 = calc_mean_binary_cross_entropy_from_probas(y_tr_M, yproba1_tr_M_l1)
    bce_valid_l1 = calc_mean_binary_cross_entropy_from_probas(y_va_N, yproba1_va_N_l1)
    print(f"L1 Penalty\n  Train BCE = {bce_train_l1}; Valid BCE = {bce_valid_l1}")

    ########################################################

    # L2 Penalty
    lr_l2 = sklearn.linear_model.LogisticRegression(penalty='l2', max_iter=max_iter, C=C, solver=solver, random_state=RANDOM_STATE)
    lr_l2.fit(x_tr_M784, y_tr_M)  # Part b

    # Record training and validation set error rate
    yproba1_tr_M_l2 = lr_l2.predict_proba(x_tr_M784)[:, 1]  # The probability of predicting class 1 on the training set
    yproba1_va_N_l2 = lr_l2.predict_proba(x_va_N784)[:, 1]  # The probability of predicting class 1 on the validation set

    error_train_l2 = sklearn.metrics.zero_one_loss(y_tr_M, yproba1_tr_M_l2 >= 0.5)
    error_valid_l2 = sklearn.metrics.zero_one_loss(y_va_N, yproba1_va_N_l2 >= 0.5)
    print(f"L2 Penalty\n  Train Error = {error_train_l2}; Valid Error = {error_valid_l2}")

    bce_train_l2 = calc_mean_binary_cross_entropy_from_probas(y_tr_M, yproba1_tr_M_l2)
    bce_valid_l2 = calc_mean_binary_cross_entropy_from_probas(y_va_N, yproba1_va_N_l2)
    print(f"L2 Penalty\n  Train BCE = {bce_train_l2}; Valid BCE = {bce_valid_l2}")

    plot_roc_curve_penalties(y_va_N, yproba1_va_N_no_penalty, yproba1_va_N_l1, yproba1_va_N_l2)


def plot_roc_curve_penalties(y_va_N, yproba1_va_N_no_penalty, yproba1_va_N_l1, yproba1_va_N_l2):
    lr_fpr_no_penalty, lr_tpr_no_penalty, _ = sklearn.metrics.roc_curve(y_va_N, yproba1_va_N_no_penalty)
    lr_fpr_l1, lr_tpr_l1, ignore_this = sklearn.metrics.roc_curve(y_va_N, yproba1_va_N_l1)
    lr_fpr_l2, lr_tpr_l2, ignore_this = sklearn.metrics.roc_curve(y_va_N, yproba1_va_N_l2)
    fig, ax = plt.subplots()

    # Set up the Error rate subplot
    ax.plot(lr_fpr_no_penalty, lr_tpr_no_penalty, 'g.-', label='Logistic Regression no penalty')
    ax.plot(lr_fpr_l1, lr_tpr_l1, 'b.-', label='Logistic Regression L1 penalty')
    ax.plot(lr_fpr_l2, lr_tpr_l2, 'r.-', label='Logistic Regression L2 penalty')
    ax.set_title('ROC Curve')
    ax.set_ylabel('TPR')
    ax.set_xlabel('FPR')
    ax.legend()
    plt.savefig('output-figures/ROC_curve_all_penalities.png')
    plt.show()


if __name__ == '__main__':
    penaltySelection()
