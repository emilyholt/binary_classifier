import os
import numpy as np
import pandas as pd

import sklearn.linear_model
import sklearn.metrics

# import plotting libraries
import matplotlib
import matplotlib.pyplot as plt

# Data loading and globals
DATA_DIR = os.path.join('data_digits_8_vs_9_noisy')  # Make sure you have downloaded data and your directory is correct

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


def lossAssessment():
    # Using sklearn.linear_model.LogisticRegression, you should fit a logistic regression models to your training split.
    C = 1e6
    solver = 'lbfgs'

    for i in range(40):
        print(i + 1)
    pass


if __name__ == "__main__":
    datasetExploration()
