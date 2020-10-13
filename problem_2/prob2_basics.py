import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sklearn.linear_model
import sklearn.metrics

DATA_DIR = os.path.join('.', 'data_sneaker_vs_sandal')

def data_exploration():
    x_test = np.loadtxt(os.path.join(DATA_DIR, 'x_test.csv'), delimiter=',', skiprows=1)
    x_test_shape = x_test.shape[0]

    # Load outcomes y arrays
    y_train = np.loadtxt(os.path.join(DATA_DIR, 'y_train.csv'), delimiter=',', skiprows=1)

    train_total = len(y_train)
    train_pos = np.count_nonzero(y_train == 1.0)
    train_frac_pos = float(train_pos) / float(train_total)
    test_percent = float(x_test_shape) / (float(train_total) + float(x_test_shape))

    print(f"Train total = {train_total}")
    print(f"train_pos total = {train_pos}")
    print(f"train_frac_pos = {train_frac_pos}")
    print(f"Test total = {x_test_shape}")
    print(f"Test percent = {test_percent}")


def split_into_train_and_valid(frac_valid=0.1428, random_state=None):
    
    if random_state is None:
        random_state = np.random
    elif type(random_state) is int:
        random_state = np.random.RandomState(random_state)

    # Load outcomes y arrays
    x_train = np.loadtxt(os.path.join(DATA_DIR, 'x_train.csv'), delimiter=',', skiprows=1)
    y_train = np.loadtxt(os.path.join(DATA_DIR, 'y_train.csv'), delimiter=',', skiprows=1)


    # Determine the number of samples that will be in the test set
    total_samples = len(y_train)
    total_positive_samples = np.count_nonzero(y_train == 1.0)
    positive_samples = np.where(y_train == 1.0)[0]
    negative_samples = np.where(y_train == 0)[0]
    n_valid_samples = int(np.ceil(frac_valid * total_samples))
    n_valid_positive_samples = n_valid_samples // 2

    # Get indices of positive samples
    positive_samples_copy = random_state.permutation(positive_samples)
    valid_positive_samples = positive_samples_copy[0:n_valid_positive_samples]
    train_positive_samples = positive_samples_copy[n_valid_positive_samples:]

    # Get indices of negative samples
    negative_samples_copy = random_state.permutation(negative_samples)
    valid_negative_samples = negative_samples_copy[0:n_valid_positive_samples]
    train_negative_samples = negative_samples_copy[n_valid_positive_samples:]

    # Get the samples from the x & y dataset to make up trainset
    x_train_set = np.vstack((x_train[train_positive_samples], x_train[train_negative_samples]))
    y_train_set = np.hstack((y_train[train_positive_samples], y_train[train_negative_samples]))

    x_valid_set = np.vstack((x_train[valid_positive_samples], x_train[valid_negative_samples]))
    y_valid_set = np.hstack((y_train[valid_positive_samples], y_train[valid_negative_samples]))

    print(f"X Train set size = {x_train_set.shape}")
    print(f"Y Train set size = {y_train_set.shape}\n")
    print(f"X Valid set size = {x_valid_set.shape}")
    print(f"Y Valid set size = {y_valid_set.shape}")
    return x_train_set, y_train_set, x_valid_set, y_valid_set


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
    plot_loss(iterations, iter_step, bce_tr_lr_i, bce_va_lr_i,)
    plot_error(iterations, iter_step, error_tr_lr_i, error_va_lr_i)

def plot_loss(iterations, iter_step, bce_tr_lr_i, bce_va_lr_i):
    fig, ax = plt.subplots()

    # Set up the Log Loss subplot
    ax.plot(np.arange(1, iterations + 1, iter_step), bce_tr_lr_i, 'b.-', label='train')
    ax.plot(np.arange(1, iterations + 1, iter_step), bce_va_lr_i, 'r.-', label='valid')
    ax.set_title('Log Loss')
    ax.set_ylabel('loss')
    ax.set_xlabel("number of iterations")
    ax.legend()

    plt.show()

def plot_error(iterations, iter_step, error_tr_lr_i):
    fig, ax = plt.subplots()

    # Set up the Error rate subplot
    ax.plot(np.arange(1, iterations + 1, iter_step), error_tr_lr_i, 'b:', label='train')
    ax.plot(np.arange(1, iterations + 1, iter_step), error_va_lr_i, 'r:', label='valid')
    ax.set_title('Error Rate')
    ax.set_ylabel('error')
    ax.set_xlabel("number of iterations")
    ax.legend()

    plt.show()

if __name__ == '__main__':
    #data_exploration()
    x_train_set, y_train_set, x_valid_set, y_valid_set = split_into_train_and_valid(frac_valid=0.1428, random_state=None)
    lossAssessment(x_train_set, y_train_set, x_valid_set, y_valid_set)