# Code stolen from dtphelan1
# Reused for problem 2

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

import sklearn.linear_model
import sklearn.metrics


def calc_mean_binary_cross_entropy_from_probas(ytrue_N, yproba1_N):
    return sklearn.metrics.log_loss(ytrue_N, yproba1_N, labels=[0, 1]) / np.log(2.0)


def plot_loss_over_c(C_grid, bce_tr_lr_i, bce_va_lr_i, filepath):
    fig, ax = plt.subplots()

    # Set up the Log Loss subplot
    x = np.log10(C_grid)
    ax.plot(x, bce_tr_lr_i, 'b.-', label='train')
    ax.plot(x, bce_va_lr_i, 'r.-', label='valid')
    ax.set_title('Log Loss at different C Values')
    ax.set_ylabel('loss')
    ax.set_xlabel("log_{10} C values")
    ax.legend()
    # plt.savefig('output-figures/LogLossCValues.png')
    plt.savefig(filepath)
    plt.show()


def plot_error_over_c(C_grid, error_tr_lr_i, error_va_lr_i, filepath):
    fig, ax = plt.subplots()

    # Set up the Error rate subplot
    x = np.log10(C_grid)
    ax.plot(x, error_tr_lr_i, 'b:', label='train')
    ax.plot(x, error_va_lr_i, 'r:', label='valid')
    ax.set_title('Error Rate at different C Values')
    ax.set_ylabel('Error Rate')
    ax.set_xlabel("log_{10} C values")
    ax.legend()
    # plt.savefig('output-figures/ErrorRateCValues.png')
    plt.savefig(filepath)
    plt.show()


def plot_roc_curve(y_va_N, yproba1_va_N, filepath):
    lr_fpr, lr_tpr, ignore_this = sklearn.metrics.roc_curve(y_va_N, yproba1_va_N)
    fig, ax = plt.subplots()

    # Set up the Error rate subplot
    ax.plot(lr_fpr, lr_tpr, 'g.-', label='Logistic Regression')
    ax.set_title('ROC Curve')
    ax.set_ylabel('TFR')
    ax.set_xlabel('TPR')
    ax.legend()
    # plt.savefig('output-figures/ROC_curve.png')
    plt.savefig(filepath)
    plt.show()


def labelDisplay(y_label):
    if (y_label == 0):
        return "Sneaker"
    elif (y_label == 1):
        return "Sandal"
    else:
        print("Unexpected label value of " % y_label)
        return "Unknown"


def rowDisplay(row):
    return str(row + 1)


def colDisplay(col):
    # 65 == 'A' in ASCII
    return chr(col + 65)


def show_images(X, y, row_ids, filepath, n_rows=3, n_cols=3):
    ''' Display images

    Args
    ----
    X : 2D array, shape (N, 784)
        Each row is a flat image vector for one example
    y : 1D array, shape (N,)
        Each row is label for one example
    row_ids : list of int
        Which rows of the dataset you want to display
    '''
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols,
        figsize=(n_cols * 3, n_rows * 3))

    for ii, row_id in enumerate(row_ids):
        cur_ax = axes.flatten()[ii]
        cur_ax.imshow(X[row_id].reshape(28, 28), interpolation='nearest', vmin=0, vmax=1, cmap='gray')
        cur_ax.set_xticks([])
        cur_ax.set_yticks([])
        row = ii // n_cols
        col = ii % n_rows
        cur_ax.set_title(f'{rowDisplay(row)}{colDisplay(col)}\nTrue Label={labelDisplay(y[row_id])}')
    plt.savefig(filepath)
    plt.show()


def analyzeMistakes(x_va_N784, y_va_N, model_path, base_filepath):
    """
    Using a saved model, analyze the FP and FN errors by plotting examples of both incorrect classifications;
    Interpretations of errors are provided in the p1.md document
    """
    # Load saved model
    with Path(model_path).open('rb') as pickle_file:
        model = pickle.load(pickle_file)
    yclass_va_N = model.predict(x_va_N784)  # The probability of predicting class 1 on the validation set

    # K = the number of incorrectly labeled classes; assumed > 0
    false_positives = np.argwhere((yclass_va_N != y_va_N) & (y_va_N == 0)).flatten()
    false_negatives = np.argwhere((yclass_va_N != y_va_N) & (y_va_N == 1)).flatten()

    n_rows = 3
    n_cols = 3

    # Sample the first row*col many error instances
    fp_displayed = false_positives[:n_cols*n_rows]
    fn_displayed = false_negatives[:n_cols*n_rows]
    # NOTE: ASSUME PNG
    fp_filepath = base_filepath.split('.png')[0] + "_fp.png"
    fn_filepath = base_filepath.split('.png')[0] + "_fn.png"
    show_images(x_va_N784, y_va_N, fp_displayed, fp_filepath, n_rows, n_cols)
    show_images(x_va_N784, y_va_N, fn_displayed, fn_filepath, n_rows, n_cols)
