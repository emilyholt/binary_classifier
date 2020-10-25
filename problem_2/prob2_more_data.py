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

from prob2_basics import *
from shared_code import *

PICKLE_ORIGINAL_MODEL = 'optimal_lrmodel.pkl'
PICKLE_AUGMENTED_MODEL = 'optimal_lrmodel_augmented.pkl'

DATA_PATH = 'augmented_data'
x_tr_M784 = np.loadtxt(os.path.join(DATA_PATH, 'x_train_set.csv'), delimiter=',')
x_va_N784 = np.loadtxt(os.path.join(DATA_PATH, 'x_valid_set.csv'), delimiter=',')

M_shape = x_tr_M784.shape
N_shape = x_va_N784.shape

N = N_shape[0]
M = M_shape[0]

y_tr_M = np.loadtxt(os.path.join(DATA_PATH, 'y_train_set.csv'), delimiter=',')
y_va_N = np.loadtxt(os.path.join(DATA_PATH, 'y_valid_set.csv'), delimiter=',')

# Load augmented data too
x_tr_M784_augmented = np.loadtxt(os.path.join(DATA_PATH, 'x_train_set_augmented.csv'), delimiter=',')
x_va_N784_augmented = np.loadtxt(os.path.join(DATA_PATH, 'x_valid_set_augmented.csv'), delimiter=',')

M_shape_augmented = x_tr_M784.shape
N_shape_augmented = x_va_N784.shape

N_augmented = N_shape_augmented[0]
M_augmented = M_shape_augmented[0]

y_tr_M_augmented = np.loadtxt(os.path.join(DATA_PATH, 'y_train_set_augmented.csv'), delimiter=',')
y_va_N_augmented = np.loadtxt(os.path.join(DATA_PATH, 'y_valid_set_augmented.csv'), delimiter=',')


def linePlot(plot, title, xlabel, ylabel, x, *lines_with_labels):
    """ A generic function for plotting a list of lines on an axis with common x and y labels
    """
    for (y_data, line_type, label) in lines_with_labels:
        plot.plot(x, y_data, line_type, label=label)
    plot.set_title(title)
    plot.set_ylabel(ylabel)
    plot.set_xlabel(xlabel)
    plot.legend()


def plotError(plot, title, x, xlabel, error_tr, error_va):
    ylabel = 'error'
    tr_line = (error_tr, 'b.-', 'train')
    va_line = (error_va, 'r.-', 'valid')
    linePlot(plot, title, xlabel, ylabel, x, tr_line, va_line)


def plotHyperparameterError(C_grid, error_tr_lr_c, error_va_lr_c):
    """ Plot Error values for training and validation sets
        as a function of different C-hyperparameters
    """
    x = np.log10(C_grid)
    xlabel = "log_{10} C"
    err_title = "Error over C-values"
    xscale = "log"
    # Set up the Error rate subplot
    plotError(plt.gca(), err_title, x, xlabel, error_tr_lr_c, error_va_lr_c)
    plt.savefig('output-figures/ErrorRateMoreData.png')
    plt.show()


def plot_roc_curve(y_va_N, yproba1_va_N, y_va_N_augmented, yproba1_va_N_augmented):
    lr_fpr, lr_tpr, ignore_this = sklearn.metrics.roc_curve(y_va_N, yproba1_va_N)
    lr_fpr_augmented, lr_tpr_augmented, ignore_this = sklearn.metrics.roc_curve(y_va_N_augmented, yproba1_va_N_augmented)
    fig, ax = plt.subplots()

    # Set up the Error rate subplot
    ax.plot(lr_fpr, lr_tpr, 'g.-', label='Logistic Regression')
    ax.plot(lr_fpr_augmented, lr_tpr_augmented, 'b.-', label='Logistic Regression with more data')
    ax.set_title('ROC Curve comparison')
    ax.set_ylabel('TPR')
    ax.set_xlabel('FPR')
    ax.legend()
    plt.savefig('output-figures/ROC_curve_moredata.png')
    plt.show()


def hyperparameterSelectionWithMoreData(plot=True, pickle_it=True):
    """ Iterate over our C_grid to see which hyperparameter gives us the best performance
    """
    C_grid = np.logspace(-9, 6, 31)
    #C_grid = np.asarray([0.01])
    solver = 'lbfgs'
    max_iter = 1000  # Fixed per instructions

    lr_models_c = list()

    error_tr_lr_c = list()
    error_va_lr_c = list()
    bce_tr_lr_i = list()
    bce_va_lr_i = list()

    for C in C_grid:
        # Build and evaluate model for this C value
        lr_c = sklearn.linear_model.LogisticRegression(max_iter=max_iter, C=C, solver=solver)
        lr_c.fit(x_tr_M784_augmented, y_tr_M_augmented)  # Part b
        lr_models_c.append(lr_c)

        # Record training and validation set error rate
        yproba1_tr_M_augmented = lr_c.predict_proba(x_tr_M784_augmented)[:, 1]  # The probability of predicting class 1 on the training set
        yproba1_va_N_augmented = lr_c.predict_proba(x_va_N784_augmented)[:, 1]  # The probability of predicting class 1 on the validation set

        error_tr_lr_c.append(sklearn.metrics.zero_one_loss(y_tr_M_augmented, yproba1_tr_M_augmented >= 0.5))
        error_va_lr_c.append(sklearn.metrics.zero_one_loss(y_va_N_augmented, yproba1_va_N_augmented >= 0.5))

        bce_tr_lr_i.append(calc_mean_binary_cross_entropy_from_probas(y_tr_M_augmented, yproba1_tr_M_augmented))
        bce_va_lr_i.append(calc_mean_binary_cross_entropy_from_probas(y_va_N_augmented, yproba1_va_N_augmented))

    # Conditionally plot the model
    if plot:
        plotHyperparameterError(C_grid, error_tr_lr_c, error_va_lr_c)
        plot_loss_over_c(C_grid, bce_tr_lr_i, bce_va_lr_i)

    min_c_index = np.argmin(error_va_lr_c)
    min_c = C_grid[min_c_index]
    print("min_c", min_c)

    chosen_model = lr_models_c[min_c_index]
    with Path(PICKLE_AUGMENTED_MODEL).open('wb') as pickle_file:
        pickle.dump(chosen_model, pickle_file)

    with Path(PICKLE_ORIGINAL_MODEL).open('rb') as pickle_file:
        model = pickle.load(pickle_file)
    yproba1_va_N = model.predict(x_va_N784)

    # Use chose model to plot metrics
    yproba1_tr_M_augmented = chosen_model.predict_proba(x_tr_M784_augmented)[:, 1]
    yproba1_va_N_augmented = chosen_model.predict_proba(x_va_N784_augmented)[:, 1]
    plot_roc_curve(y_va_N, yproba1_va_N, y_va_N_augmented, yproba1_va_N_augmented)


if __name__ == '__main__':
    hyperparameterSelectionWithMoreData(plot=False, pickle_it=True)
