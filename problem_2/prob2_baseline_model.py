import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle

import sklearn.linear_model
import sklearn.metrics


# This part stolen from dtphelan1
PICKLE_DIR = "output_models"
PICKLE_ORIGINAL_MODEL = 'baseline_optimal_lrmodel.pkl'

DATA_PATH = 'data_sneaker_vs_sandal'
x_tr_M784 = np.loadtxt(os.path.join(DATA_PATH, 'x_train_set.csv'), delimiter=',')
x_va_N784 = np.loadtxt(os.path.join(DATA_PATH, 'x_valid_set.csv'), delimiter=',')

M_shape = x_tr_M784.shape
N_shape = x_va_N784.shape

N = N_shape[0]
M = M_shape[0]

y_tr_M = np.loadtxt(os.path.join(DATA_PATH, 'y_train_set.csv'), delimiter=',')
y_va_N = np.loadtxt(os.path.join(DATA_PATH, 'y_valid_set.csv'), delimiter=',')


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
    # Set up the Error rate subplot
    plotError(plt.gca(), err_title, x, xlabel, error_tr_lr_c, error_va_lr_c)
    plt.show()


def hyperparameterSelection(plot=False, pickle_it=True):
    """ Iterate over our C_grid to see which hyperparameter gives us the best performance
    """
    C_grid = np.logspace(-9, 6, 31)
    # NOTE: This was the optimal C-value previously; saving for posterity
    # C_grid = np.asarray([0.31622776601683794])
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
        lr_c.fit(x_tr_M784, y_tr_M)  # Part b
        lr_models_c.append(lr_c)

        # Record training and validation set error rate
        yproba1_tr_M = lr_c.predict_proba(x_tr_M784)[:, 1]  # The probability of predicting class 1 on the training set
        yproba1_va_N = lr_c.predict_proba(x_va_N784)[:, 1]  # The probability of predicting class 1 on the validation set

        error_tr_lr_c.append(sklearn.metrics.zero_one_loss(y_tr_M, yproba1_tr_M >= 0.5))
        error_va_lr_c.append(sklearn.metrics.zero_one_loss(y_va_N, yproba1_va_N >= 0.5))

        bce_tr_lr_i.append(calc_mean_binary_cross_entropy_from_probas(y_tr_M, yproba1_tr_M))
        bce_va_lr_i.append(calc_mean_binary_cross_entropy_from_probas(y_va_N, yproba1_va_N))

    # Conditionally plot the model
    if plot:
        plotHyperparameterError(C_grid, error_tr_lr_c, error_va_lr_c)
        plot_loss_over_c(C_grid, bce_tr_lr_i, bce_va_lr_i)

    min_c_index = np.argmin(error_va_lr_c)
    min_c = C_grid[min_c_index]
    print("min_c", min_c)

    chosen_model = lr_models_c[min_c_index]
    with Path(os.path.join(PICKLE_DIR, PICKLE_ORIGINAL_MODEL)).open('wb') as pickle_file:
        pickle.dump(chosen_model, pickle_file)
    return chosen_model


def plot_roc_curve(y_va_N, yproba1_va_N):
    lr_fpr, lr_tpr, _ = sklearn.metrics.roc_curve(y_va_N, yproba1_va_N)
    fig, ax = plt.subplots()

    # Set up the Error rate subplot
    ax.plot(lr_fpr, lr_tpr, 'g.-', label='Logistic Regression')
    ax.set_title('ROC Curve')
    ax.set_ylabel('TFR')
    ax.set_xlabel('TPR')
    ax.legend()
    plt.savefig('output-figures/ROC_curve_baseline.png')
    plt.show()


if __name__ == '__main__':
    split_into_train_and_valid(n_valid_samples=2000, random_state=random_state)
    optimal_model = hyperparameterSelection(plot=True, pickle_it=True)
    yproba1_tr_M = optimal_model.predict_proba(x_tr_M784)[:, 1]
    yproba1_va_N = optimal_model.predict_proba(x_va_N784)[:, 1]
    plot_roc_curve(y_va_N, yproba1_va_N)
