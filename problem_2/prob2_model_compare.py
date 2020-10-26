# KNN Classifier code

import os
from pathlib import Path
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.metrics
import sklearn.neighbors

from shared_code import calc_mean_binary_cross_entropy_from_probas

# This part stolen from dtphelan1
PICKLE_DIR = "output_models"
PICKLE_BASELINE_MODEL = 'optimal_lrmodel.pkl'
PICKLE_AUGMENTED_MODEL = 'optimal_lrmodel_augmented.pkl'
PICKLE_KNN_MODEL = 'optimal_knn_model.pkl'

PICKLE_BASELINE_FILEPATH = os.path.join(PICKLE_DIR, PICKLE_BASELINE_MODEL)
PICKLE_AUGMENTED_FILEPATH = os.path.join(PICKLE_DIR, PICKLE_AUGMENTED_MODEL)
PICKLE_KNN_FILEPATH = os.path.join(PICKLE_DIR, PICKLE_KNN_MODEL)


DATA_PATH = 'data_sneaker_vs_sandal'
x_tr_M784 = np.loadtxt(os.path.join(DATA_PATH, 'x_train_set.csv'), delimiter=',')
x_va_N784 = np.loadtxt(os.path.join(DATA_PATH, 'x_valid_set.csv'), delimiter=',')

y_tr_M = np.loadtxt(os.path.join(DATA_PATH, 'y_train_set.csv'), delimiter=',')
y_va_N = np.loadtxt(os.path.join(DATA_PATH, 'y_valid_set.csv'), delimiter=',')

# Load augmented data too
x_tr_M784_augmented = np.loadtxt(os.path.join(DATA_PATH, 'x_train_set_augmented.csv'), delimiter=',')
x_va_N784_augmented = np.loadtxt(os.path.join(DATA_PATH, 'x_valid_set_augmented.csv'), delimiter=',')

y_tr_M_augmented = np.loadtxt(os.path.join(DATA_PATH, 'y_train_set_augmented.csv'), delimiter=',')
y_va_N_augmented = np.loadtxt(os.path.join(DATA_PATH, 'y_valid_set_augmented.csv'), delimiter=',')


def model_comparison():

    # Load saved baseline model
    with Path(PICKLE_BASELINE_FILEPATH).open('rb') as pickle_file:
        model = pickle.load(pickle_file)
    yclass_tr_M_baseline = model.predict_proba(x_tr_M784)[:, 1]
    yclass_va_N_baseline = model.predict_proba(x_va_N784)[:, 1]
    baseline_train_error = calc_mean_binary_cross_entropy_from_probas(y_tr_M, yclass_tr_M_baseline)
    baseline_valid_error = calc_mean_binary_cross_entropy_from_probas(y_va_N, yclass_va_N_baseline)
    print(f"Baseline train BCE = {baseline_train_error}; Baseline Valid BCE = {baseline_valid_error}")

    # Load saved augmented model
    with Path(PICKLE_AUGMENTED_FILEPATH).open('rb') as pickle_file:
        model = pickle.load(pickle_file)
    yclass_tr_M_augmented = model.predict_proba(x_tr_M784_augmented)[:, 1]
    yclass_va_N_augmented = model.predict_proba(x_va_N784_augmented)[:, 1]
    augmented_train_error = calc_mean_binary_cross_entropy_from_probas(y_tr_M_augmented, yclass_tr_M_augmented)
    augmented_valid_error = calc_mean_binary_cross_entropy_from_probas(y_va_N_augmented, yclass_va_N_augmented)
    print(f"Augmented train BCE = {augmented_train_error}; augmented Valid BCE = {augmented_valid_error}")

    # Load saved knn model
    with Path(PICKLE_KNN_FILEPATH).open('rb') as pickle_file:
        model = pickle.load(pickle_file)
    yclass_tr_M_knn = model.predict_proba(x_tr_M784)[:, 1]
    yclass_va_N_knn = model.predict_proba(x_va_N784)[:, 1]
    knn_train_error = calc_mean_binary_cross_entropy_from_probas(y_tr_M, yclass_tr_M_knn)
    knn_valid_error = calc_mean_binary_cross_entropy_from_probas(y_va_N, yclass_va_N_knn)
    print(f"knn train BCE = {knn_train_error}; knn Valid BCE = {knn_valid_error}")

    # Plot ROC curve
    plot_train_comparison_roc_curve(yclass_tr_M_baseline, yclass_tr_M_augmented, yclass_tr_M_knn)
    plot_valid_comparison_roc_curve(yclass_va_N_baseline, yclass_va_N_augmented, yclass_va_N_knn)


def plot_train_comparison_roc_curve(yclass_tr_M_baseline, yclass_tr_M_augmented, yclass_tr_M_knn):
    lr_fpr_baseline, lr_tpr_baseline, ignore_this = sklearn.metrics.roc_curve(y_tr_M, yclass_tr_M_baseline)
    lr_fpr_augmented, lr_tpr_augmented, ignore_this = sklearn.metrics.roc_curve(y_tr_M_augmented, yclass_tr_M_augmented)
    lr_fpr_knn, lr_tpr_knn, ignore_this = sklearn.metrics.roc_curve(y_tr_M, yclass_tr_M_knn)
    fig, ax = plt.subplots()

    # Set up the Error rate subplot
    ax.plot(lr_fpr_baseline, lr_tpr_baseline, 'g.-', label='Baseline Classifier')
    ax.plot(lr_fpr_augmented, lr_tpr_augmented, 'r.-', label='Augmented-Data Classifier')
    ax.plot(lr_fpr_knn, lr_tpr_knn, 'b.-', label='KNN Classifier')
    ax.set_title('ROC Curve')
    ax.set_ylabel('TPR')
    ax.set_xlabel('FPR')
    ax.legend()
    plt.savefig('output-figures/ROC_curve_train_comparison.png')
    plt.show()


def plot_valid_comparison_roc_curve(yclass_va_N_baseline, yclass_va_N_augmented, yclass_va_N_knn):
    lr_fpr_baseline, lr_tpr_baseline, ignore_this = sklearn.metrics.roc_curve(y_va_N, yclass_va_N_baseline)
    lr_fpr_augmented, lr_tpr_augmented, ignore_this = sklearn.metrics.roc_curve(y_va_N_augmented, yclass_va_N_augmented)
    lr_fpr_knn, lr_tpr_knn, ignore_this = sklearn.metrics.roc_curve(y_va_N, yclass_va_N_knn)
    fig, ax = plt.subplots()

    # Set up the Error rate subplot
    ax.plot(lr_fpr_baseline, lr_tpr_baseline, 'g.-', label='Baseline Classifier')
    ax.plot(lr_fpr_augmented, lr_tpr_augmented, 'r.-', label='Augmented-Data Classifier')
    ax.plot(lr_fpr_knn, lr_tpr_knn, 'b.-', label='KNN Classifier')
    ax.set_title('ROC Curve')
    ax.set_ylabel('TPR')
    ax.set_xlabel('FPR')
    ax.legend()
    plt.savefig('output-figures/ROC_curve_valid_comparison.png')
    plt.show()


if __name__ == '__main__':
    model_comparison()
