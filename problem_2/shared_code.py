# Code stolen from dtphelan1
# Reused for problem 2

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
