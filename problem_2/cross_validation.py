import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from shared_code import *

DATA_DIR = 'data_sneaker_vs_sandal'
RANDOM_STATE = 1234
x_tr_M784_augmented = np.loadtxt(os.path.join(DATA_DIR, 'x_train_augmented.csv'), delimiter=',',  skiprows=1)

y_tr_M_augmented = np.loadtxt(os.path.join(DATA_DIR, 'y_train_augmented.csv'), delimiter=',',  skiprows=1)

C_grid = np.logspace(-9, 6, 31)

def hyperparameterSelection(x_train_fold, y_train_fold, x_valid_fold, y_valid_fold):
    """ Iterate over our C_grid to see which hyperparameter gives us the best performance
    """
    
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
        lr_c.fit(x_train_fold, y_train_fold)  # Part b
        lr_models_c.append(lr_c)

        # Record training and validation set error rate
        yproba1_tr_M = lr_c.predict_proba(x_train_fold)[:, 1]  # The probability of predicting class 1 on the training set
        yproba1_va_N = lr_c.predict_proba(x_valid_fold)[:, 1]  # The probability of predicting class 1 on the validation set

        error_tr_lr_c.append(sklearn.metrics.zero_one_loss(y_train_fold, yproba1_tr_M >= 0.5))
        error_va_lr_c.append(sklearn.metrics.zero_one_loss(y_valid_fold, yproba1_va_N >= 0.5))

        bce_tr_lr_i.append(calc_mean_binary_cross_entropy_from_probas(y_train_fold, yproba1_tr_M))
        bce_va_lr_i.append(calc_mean_binary_cross_entropy_from_probas(y_valid_fold, yproba1_va_N))

    min_c_index = np.argmin(error_va_lr_c)
    min_c = C_grid[min_c_index]
    print(f"\n\n min_c = {min_c}\n\n")

    # Use chose model to plot metrics
    # chosen_model = lr_models_c[min_c_index]
    # yproba1_tr_M = chosen_model.predict_proba(x_train_fold)[:, 1]
    # yproba1_va_N = chosen_model.predict_proba(x_valid_fold)[:, 1]

    return bce_tr_lr_i, bce_va_lr_i


def split_into_train_and_valid_folds(x, y, id=None, n_valid_samples=2000, random_state=None):
    """
    Given an x and y dataset, split it into two train-validation sets,
    each named with the `id` value to distinguish them from other similar datasets
    """

    if random_state is None:
        random_state = np.random
    elif type(random_state) is int:
        random_state = np.random.RandomState(random_state)

    # Determine the number of samples that will be in the test set
    total_samples = len(y)
    total_positive_samples = np.count_nonzero(y == 1.0)
    positive_samples = np.where(y == 1.0)[0]
    negative_samples = np.where(y == 0)[0]
    n_valid_positive_samples = n_valid_samples // 2

    # Split data into folds
    x_train_set_folds = []
    y_train_set_folds = []

    x_valid_set_folds = []
    y_valid_set_folds = []

    bce_tr_lr_fold = list()
    bce_va_lr_fold = list()

    for ind in range(6):
        positive_samples_copy = random_state.permutation(positive_samples)
        start_ind = ind * n_valid_positive_samples
        stop_ind = (ind + 1) * n_valid_positive_samples
        valid_positive_samples = positive_samples_copy[start_ind:stop_ind]
        # train_positive_samples = positive_samples_copy[0:start_ind] + positive_samples_copy[stop_ind:]
        train_positive_samples = [f for f in positive_samples_copy if f not in valid_positive_samples]

        # Get indices of negative samples
        negative_samples_copy = random_state.permutation(negative_samples)
        valid_negative_samples = negative_samples_copy[start_ind:stop_ind]
        # train_negative_samples = negative_samples_copy[0:start_ind] + positive_samples_copy[stop_ind:]
        train_negative_samples = [f for f in negative_samples_copy if f not in valid_negative_samples]

        # Get the samples from the x & y dataset to make up trainset
        x_train_set = np.vstack((x[train_positive_samples], x[train_negative_samples]))
        y_train_set = np.hstack((y[train_positive_samples], y[train_negative_samples]))

        x_valid_set = np.vstack((x[valid_positive_samples], x[valid_negative_samples]))
        y_valid_set = np.hstack((y[valid_positive_samples], y[valid_negative_samples]))

        bce_tr_lr, bce_va_lr = hyperparameterSelection(x_train_set, y_train_set, x_valid_set, y_valid_set)
        bce_tr_lr_fold.append(bce_tr_lr)
        bce_va_lr_fold.append(bce_va_lr)
    
    return bce_tr_lr_fold, bce_va_lr_fold

def plot_train_cv_logloss(bce_tr_lr_fold):
    fig, ax = plt.subplots()

    # Set up the Log Loss subplot
    x = np.log10(C_grid)
    ax.plot(x, bce_tr_lr_fold[0], 'b.-', label='Fold 1')
    ax.plot(x, bce_tr_lr_fold[1], 'g.-', label='Fold 2')
    ax.plot(x, bce_tr_lr_fold[2], 'r.-', label='Fold 3')
    ax.plot(x, bce_tr_lr_fold[3], 'c.-', label='Fold 4')
    ax.plot(x, bce_tr_lr_fold[4], 'm.-', label='Fold 5')
    ax.plot(x, bce_tr_lr_fold[5], 'y.-', label='Fold 6')
    ax.set_title('Log Loss over 6-fold Cross Validation on Train sets')
    ax.set_ylabel('Log loss')
    ax.set_xlabel("log_{10} C")
    ax.legend()
    plt.savefig('output-figures/CrossValidationTrainset.png')
    plt.show() 

def plot_valid_cv_logloss(bce_va_lr_fold):
    fig, ax = plt.subplots()

    # Set up the Log Loss subplot
    x = np.log10(C_grid)
    ax.plot(x, bce_va_lr_fold[0], 'b.-', label='Fold 1')
    ax.plot(x, bce_va_lr_fold[1], 'g.-', label='Fold 2')
    ax.plot(x, bce_va_lr_fold[2], 'r.-', label='Fold 3')
    ax.plot(x, bce_va_lr_fold[3], 'c.-', label='Fold 4')
    ax.plot(x, bce_va_lr_fold[4], 'm.-', label='Fold 5')
    ax.plot(x, bce_va_lr_fold[5], 'y.-', label='Fold 6')
    ax.set_title('Log Loss over 6-fold Cross Validation on Holdout sets')
    ax.set_ylabel('Log loss')
    ax.set_xlabel("log_{10} C")
    ax.legend()
    plt.savefig('output-figures/CrossValidationValidset.png')
    plt.show()   


if __name__ == '__main__':
    # Load outcomes y arrays
    x_train = np.loadtxt(os.path.join(DATA_DIR, 'x_train.csv'), delimiter=',', skiprows=1)
    y_train = np.loadtxt(os.path.join(DATA_DIR, 'y_train.csv'), delimiter=',', skiprows=1)
    # data_exploration()
    # For consistent splits, use a random_state param
    bce_tr_lr_fold, bce_va_lr_fold = split_into_train_and_valid_folds(x_train, y_train, None, n_valid_samples=2000, random_state=RANDOM_STATE)
    plot_train_cv_logloss(bce_tr_lr_fold)
    plot_valid_cv_logloss(bce_va_lr_fold)
