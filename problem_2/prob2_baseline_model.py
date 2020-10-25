import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle

import sklearn.linear_model
import sklearn.metrics

from shared_code import calc_mean_binary_cross_entropy_from_probas, plot_loss_over_c, plot_error_over_c, plot_roc_curve
from prob2_dataset_splitting import split_into_train_and_valid


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
        plot_error_over_c(C_grid, error_tr_lr_c, error_va_lr_c, 'output-figures/ErrorRateCValues_baseline.png')
        plot_loss_over_c(C_grid, bce_tr_lr_i, bce_va_lr_i, 'output-figures/LogLossCValues_baseline.png')

    min_c_index = np.argmin(error_va_lr_c)
    min_c = C_grid[min_c_index]
    print("min_c", min_c)

    chosen_model = lr_models_c[min_c_index]
    with Path(os.path.join(PICKLE_DIR, PICKLE_ORIGINAL_MODEL)).open('wb') as pickle_file:
        pickle.dump(chosen_model, pickle_file)
    return chosen_model


if __name__ == '__main__':
    # For consistent splits, use a random_state param
    random_state = 100
    split_into_train_and_valid(n_valid_samples=2000, random_state=random_state)
    optimal_model = hyperparameterSelection(plot=True, pickle_it=True)
    yproba1_tr_M = optimal_model.predict_proba(x_tr_M784)[:, 1]
    yproba1_va_N = optimal_model.predict_proba(x_va_N784)[:, 1]
    plot_roc_curve(y_va_N, yproba1_va_N, 'output-figures/ROC_curve_baseline.png')
    print("Finished training, then selecting optimal baseline model and plotting related figures")
