# KNN Classifier code

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.metrics
import sklearn.neighbors

from prob2_basics import *
from shared_code import *


def knn_classifier(x_tr_M784, y_tr_M, x_va_N784, y_va_N):
    solver = 'lbfgs'
    max_neighbors = 11
    neighbor_step = 2

    error_tr_lr_i = list()
    error_va_lr_i = list()

    bce_tr_lr_i = list()
    bce_va_lr_i = list()
    neighbors = list(range(1, max_neighbors, neighbor_step))

    for ind in neighbors:
        print(f"Fitting KNN classifier with {ind} neighbors")
        knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=ind)
        knn.fit(x_tr_M784, y_tr_M)  # Part b

        yproba1_tr_M = knn.predict_proba(x_tr_M784)[:, 1]  # The probability of predicting class 1 on the training set
        yproba1_va_N = knn.predict_proba(x_va_N784)[:, 1]  # The probability of predicting class 1 on the validation set

        error_tr_lr_i.append(sklearn.metrics.zero_one_loss(y_tr_M, yproba1_tr_M >= 0.5))
        error_va_lr_i.append(sklearn.metrics.zero_one_loss(y_va_N, yproba1_va_N >= 0.5))

        bce_tr_lr_i.append(calc_mean_binary_cross_entropy_from_probas(y_tr_M, yproba1_tr_M))
        bce_va_lr_i.append(calc_mean_binary_cross_entropy_from_probas(y_va_N, yproba1_va_N))
    
    # Plot error & losses
    plot_loss_over_neighbors(neighbors, bce_tr_lr_i, bce_va_lr_i,)
    plot_error_over_neighbors(neighbors, error_tr_lr_i, error_va_lr_i)

    # Plot ROC curve
    plot_roc_curve(y_va_N, yproba1_va_N)

    # Find optimal number of neighbors with best bce & error rates
    best_error_ind = error_va_lr_i.index(min(error_va_lr_i))
    best_bce_ind = bce_va_lr_i.index(min(bce_va_lr_i))
    print(f"number of neighbors with lowest error = {neighbors[best_error_ind]}")
    print(f"number of neighbors with lowest bce = {neighbors[best_bce_ind]}")


def plot_loss_over_neighbors(neighbors, bce_tr_lr_i, bce_va_lr_i):
    fig, ax = plt.subplots()

    # Set up the Log Loss subplot
    ax.plot(neighbors, bce_tr_lr_i, 'b.-', label='train')
    ax.plot(neighbors, bce_va_lr_i, 'r.-', label='valid')
    ax.set_title('Log Loss over neighbors')
    ax.set_ylabel('loss')
    ax.set_xlabel("Number of neighbors")
    ax.legend()
    plt.savefig('LogLossNeighbors.png')
    plt.show()


def plot_error_over_neighbors(neighbors, error_tr_lr_i, error_va_lr_i):
    fig, ax = plt.subplots()

    # Set up the Error rate subplot
    ax.plot(neighbors, error_tr_lr_i, 'b:', label='train')
    ax.plot(neighbors, error_va_lr_i, 'r:', label='valid')
    ax.set_title('Error Rate over neighbors')
    ax.set_ylabel('error')
    ax.set_xlabel("number of neighbors")
    ax.legend()
    plt.savefig('ErrorRateNeighbors.png')
    plt.show()

if __name__ == '__main__':
    data_exploration()
    x_train_set, y_train_set, x_valid_set, y_valid_set = split_into_train_and_valid(n_valid_samples=2000, random_state=None)
    knn_classifier(x_train_set, y_train_set, x_valid_set, y_valid_set)
    