import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.metrics
import sklearn.neighbors
import sklearn.pipeline

from prob2_basics import *
from shared_code import *

x_tr_M784 = np.loadtxt('x_train_set.csv', delimiter=',')
x_va_N784 = np.loadtxt('x_valid_set.csv', delimiter=',')

M_shape = x_tr_M784.shape
N_shape = x_va_N784.shape

N = N_shape[0]
M = M_shape[0]

y_tr_M = np.loadtxt('y_train_set.csv', delimiter=',')
y_va_N = np.loadtxt('y_valid_set.csv', delimiter=',')

def poly_features_expr():

    # Using sklearn.linear_model.LogisticRegression, you should fit a logistic regression models to your training split.
    C = 0.31622776601683794
    solver = 'lbfgs'
    iterations = 1000

    lr_models = list()
    error_tr_pipeline_i = list()
    error_va_pipeline_i = list()

    bce_tr_pipeline_i = list()
    bce_va_pipeline_i = list()

    poly_degrees = [1, 2]

    for ii, degree in enumerate(poly_degrees):
        
        pipeline = sklearn.pipeline.Pipeline([
            ("step1", sklearn.preprocessing.PolynomialFeatures(degree)), # create custom Poly featurizer
            ("step2", sklearn.linear_model.LogisticRegression(max_iter=iterations, C=C, solver=solver)),
            ])

        # Train the model
        print(f"Fitting model with poly degree {degree}")
        pipeline.fit(x_tr_M784, y_tr_M)
        lr_models.append(pipeline)

        yproba1_tr_M = pipeline.predict_proba(x_tr_M784)[:, 1]  # The probability of predicting class 1 on the training set
        yproba1_va_N = pipeline.predict_proba(x_va_N784)[:, 1]  # The probability of predicting class 1 on the validation set

        error_tr_pipeline_i.append(sklearn.metrics.zero_one_loss(y_tr_M, yproba1_tr_M >= 0.5))
        error_va_pipeline_i.append(sklearn.metrics.zero_one_loss(y_va_N, yproba1_va_N >= 0.5))

        bce_tr_pipeline_i.append(calc_mean_binary_cross_entropy_from_probas(y_tr_M, yproba1_tr_M))
        bce_va_pipeline_i.append(calc_mean_binary_cross_entropy_from_probas(y_va_N, yproba1_va_N))
    
        print("degree %3d | train error % 10.3f | valid error %10.3f" % (
            degree, error_tr_pipeline_i[ii], error_va_pipeline_i[ii]))
    
    plot_loss_over_poly_degrees(poly_degrees, bce_tr_pipeline_i, bce_va_pipeline_i,)
    plot_error_over_poly_degrees(poly_degrees, error_tr_pipeline_i, error_va_pipeline_i)

    # Plot ROC curve
    plot_roc_curve(y_va_N, yproba1_va_N)

    # Find C value with best bce & error rates
    best_error_ind = error_va_pipeline_i.index(min(error_va_pipeline_i))
    print(f"Error values = {error_va_pipeline_i}")

    best_bce_ind = bce_va_pipeline_i.index(min(bce_va_pipeline_i))
    print(f"BCE values = {bce_va_pipeline_i}")
    
    print(f"poly degree with lowest error = {best_error_ind + 1}")
    print(f"poly degree with lowest bce = {best_bce_ind + 1}")

    chosen_model = lr_models[best_bce_ind]
    with Path(PICKLE_ORIGINAL_MODEL).open('wb') as pickle_file:
        pickle.dump(chosen_model, pickle_file)

    yproba1_tr_M = chosen_model.predict_proba(x_tr_M784)[:, 1]
    yproba1_va_N = chosen_model.predict_proba(x_va_N784)[:, 1]
    plot_roc_curve(y_va_N, yproba1_va_N)

    

def plot_loss_over_poly_degrees(poly_degrees, bce_tr_pipeline_i, bce_va_pipeline_i):
    fig, ax = plt.subplots()

    # Set up the Log Loss subplot
    ax.plot(poly_degrees, bce_tr_pipeline_i, 'b.-', label='train')
    ax.plot(poly_degrees, bce_va_pipeline_i, 'r.-', label='valid')
    ax.set_title('Log Loss over polynomial degree transforms')
    ax.set_ylabel('loss')
    ax.set_xlabel("Polynomial degree")
    ax.legend()
    plt.savefig('LogLossPolynomialDegrees.png')
    plt.show()

def plot_error_over_poly_degrees(poly_degrees, error_tr_pipeline_i, error_va_pipeline_i):
    fig, ax = plt.subplots()

    # Set up the Error rate subplot
    ax.plot(poly_degrees, error_tr_pipeline_i, 'b:', label='train')
    ax.plot(poly_degrees, error_va_pipeline_i, 'r:', label='valid')
    ax.set_title('Error Rate over polynomial degree transforms')
    ax.set_ylabel('Error')
    ax.set_xlabel("Polynomial degree")
    ax.legend()
    plt.savefig('ErrorRatePolynomialDegrees.png')
    plt.show()

if __name__ == '__main__':
    # data_exploration()
    # x_train_set, y_train_set, x_valid_set, y_valid_set = split_into_train_and_valid(n_valid_samples=2000, random_state=None)
    poly_features_expr()

