from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import sklearn.linear_model
import sklearn.metrics

from helpers import calc_mean_binary_cross_entropy_from_probas
from show_images import show_images


# Data loading and globals
path, filename = os.path.split(__file__)
DATA_DIR = os.path.join(path, 'data_digits_8_vs_9_noisy')  # Make sure you have downloaded data and your directory is correct
PICKLE_MODEL = os.path.join(DATA_DIR, 'optimal_lrmodel.pkl')
FALSE_POSITIVES_FILE = "false_positives_indices.txt"
FALSE_NEGATIVES_FILE = "false_negatives_indices.txt"

x_tr_M784 = np.loadtxt(os.path.join(DATA_DIR, 'x_train.csv'), delimiter=',', skiprows=1)
x_va_N784 = np.loadtxt(os.path.join(DATA_DIR, 'x_valid.csv'), delimiter=',', skiprows=1)
x_te_N784 = np.loadtxt(os.path.join(DATA_DIR, 'x_test.csv'), delimiter=',', skiprows=1)

M_shape = x_tr_M784.shape
N_shape = x_va_N784.shape

N = N_shape[0]
M = M_shape[0]

y_tr_M = np.loadtxt(os.path.join(DATA_DIR, 'y_train.csv'), delimiter=',', skiprows=1)
y_va_N = np.loadtxt(os.path.join(DATA_DIR, 'y_valid.csv'), delimiter=',', skiprows=1)


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


def lossErrorAssessment():
    # Using sklearn.linear_model.LogisticRegression, you should fit a logistic regression models to your training split.
    C = 1e6
    solver = 'lbfgs'
    iterations = 40
    iter_step = 1

    error_tr_lr_i = list()
    error_va_lr_i = list()

    bce_tr_lr_i = list()
    bce_va_lr_i = list()

    for i in range(0, iterations, iter_step):
        lr_i = sklearn.linear_model.LogisticRegression(max_iter=i+1, C=C, solver=solver)
        lr_i.fit(x_tr_M784, y_tr_M)  # Part b

        yproba1_tr_M = lr_i.predict_proba(x_tr_M784)[:, 1]  # The probability of predicting class 1 on the training set
        yproba1_va_N = lr_i.predict_proba(x_va_N784)[:, 1]  # The probability of predicting class 1 on the validation set

        error_tr_lr_i.append(sklearn.metrics.zero_one_loss(y_tr_M, yproba1_tr_M >= 0.5))
        error_va_lr_i.append(sklearn.metrics.zero_one_loss(y_va_N, yproba1_va_N >= 0.5))

        bce_tr_lr_i.append(calc_mean_binary_cross_entropy_from_probas(y_tr_M, yproba1_tr_M))
        bce_va_lr_i.append(calc_mean_binary_cross_entropy_from_probas(y_va_N, yproba1_va_N))
    plotLossErrorOverIterations(iterations, iter_step, error_tr_lr_i, error_va_lr_i, bce_tr_lr_i, bce_va_lr_i,)


def linePlot(plot, title, xlabel, ylabel, x, *lines_with_labels):
    """ A generic function for plotting a list of lines on an axis with common x and y labels
    """
    for (y_data, line_type, label) in lines_with_labels:
        plot.plot(x, y_data, line_type, label=label)
    plot.set_title(title)
    plot.set_ylabel(ylabel)
    plot.set_xlabel(xlabel)
    plot.legend()


def plotLoss(plot, title, x, xlabel, bce_tr, bce_va):
    ylabel = 'loss'
    # y_data, line_type, label is the shape of a line tuple
    tr_line = (bce_tr, 'b.-', 'train')
    va_line = (bce_va, 'r.-', 'valid')
    linePlot(plot, title, xlabel, ylabel, x, tr_line, va_line)


def plotError(plot, title, x, xlabel, error_tr, error_va):
    ylabel = 'error'
    tr_line = (error_tr, 'b.-', 'train')
    va_line = (error_va, 'r.-', 'valid')
    linePlot(plot, title, xlabel, ylabel, x, tr_line, va_line)


def plotLossErrorOverIterations(iterations, iter_step, error_tr_lr_i, error_va_lr_i, bce_tr_lr_i, bce_va_lr_i,):
    """ Graph two subplots, visualizing Log-Loss and Error,
        as a function of different max_iteration values over training and validation
    """
    _, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=False)
    x = np.arange(1, iterations + 1, iter_step)
    xlabel = 'number of iterations'
    # Set up the Log Loss subplot
    log_loss_title = "Log Loss over Training Iterations"
    plotLoss(axes[0], log_loss_title, x, xlabel, bce_tr_lr_i, bce_va_lr_i)
    # Set up the Error rate subplot
    err_title = "Error over Training Iterations"
    plotError(axes[1], err_title, x, xlabel, error_tr_lr_i, error_va_lr_i)
    plt.show()


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
    plt.show()


def hyperparameterSelection(plot=False, pickle_it=True):
    """ Iterate over our C_grid to see which hyperparameter gives us the best performance
    """
    # C_grid = np.logspace(-9, 6, 31)
    C_grid = np.asarray([0.01])
    solver = 'lbfgs'
    max_iter = 1000  # Fixed per instructions

    lr_models_c = list()

    error_tr_lr_c = list()
    error_va_lr_c = list()

    for C in C_grid:
        # Build and evaluate model for this C value
        lr_c = sklearn.linear_model.LogisticRegression(max_iter=max_iter, C=C, solver=solver)
        lr_c.fit(x_tr_M784, y_tr_M)  # Part b
        lr_models_c.append(lr_c)

        # Record training and validation set error rate
        yproba1_tr_M = lr_c.predict_proba(x_tr_M784)[:, 1]  # The probability of predicting class 1 on the training set
        yproba1_va_N = lr_c.predict_proba(x_va_N784)[:, 1]  # The probability of predicting class 1 on the validation set

        error_tr_lr_c.append(sklearn.metrics.zero_one_loss(y_tr_M, yproba1_tr_M >= 0.5))
        error_va_lr_c.append(sklearn.metrics.zero_one_loss(y_va_N, yproba1_va_N >= 0.5))

    # Conditionally plot the model
    if plot:
        plotHyperparameterError(C_grid, error_tr_lr_c, error_va_lr_c)

    min_c_index = np.argmin(error_va_lr_c)
    min_c = C_grid[min_c_index]
    print("min_c", min_c)

    if pickle_it:
        with Path(PICKLE_MODEL).open('wb') as pickle_file:
            chosen_model = lr_models_c[min_c_index]
            pickle.dump(chosen_model, pickle_file)


def analyzeErrors(saveIndices=False):
    """
    Using a saved model, write to disk
    """
    # Load saved model
    with Path(PICKLE_MODEL).open('rb') as pickle_file:
        model = pickle.load(pickle_file)
    yclass_va_N = model.predict(x_va_N784)  # The probability of predicting class 1 on the validation set

    # K = the number of incorrectly labeled classes; assumed > 0
    false_positives = np.argwhere((yclass_va_N != y_va_N) & (y_va_N == 0)).flatten()
    print(yclass_va_N[false_positives])
    false_negatives = np.argwhere((yclass_va_N != y_va_N) & (y_va_N == 1)).flatten()
    print(yclass_va_N[false_negatives])

    # Save the indices for running the show_images script independently
    if (saveIndices):
        np.savetxt(os.path.join(DATA_DIR, FALSE_POSITIVES_FILE), false_positives, fmt="%d")
        np.savetxt(os.path.join(DATA_DIR, FALSE_NEGATIVES_FILE), false_negatives, fmt="%d")
    n_rows = 3
    n_cols = 3

    # Sample the first row*col many error instances
    fp_displayed = false_positives[:n_cols*n_rows]
    fn_displayed = false_negatives[:n_cols*n_rows]
    show_images(x_va_N784, y_va_N, fp_displayed, n_rows, n_cols)
    show_images(x_va_N784, y_va_N, fn_displayed, n_rows, n_cols)


if __name__ == "__main__":
    # datasetExploration()
    # lossErrorAssessment()
    # hyperparameterSelection(plot=False, pickle_it=True)
    analyzeErrors()
