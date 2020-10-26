import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle

import sklearn.linear_model
import sklearn.metrics

from shared_code import (
    calc_mean_binary_cross_entropy_from_probas,
    plot_loss_over_c,
    plot_error_over_c,
    plot_roc_curve,
    analyzeMistakes,
    runModelOnTest
)
from prob2_dataset_splitting import split_into_train_and_valid

RANDOM_STATE = 1234

# Where to load the incoming data from
DATA_DIR = 'data_sneaker_vs_sandal'
x_tr_M784 = np.loadtxt(os.path.join(DATA_DIR, 'x_train_set.csv'), delimiter=',')
x_va_N784 = np.loadtxt(os.path.join(DATA_DIR, 'x_valid_set.csv'), delimiter=',')
x_te_N784 = np.loadtxt(os.path.join(DATA_DIR, 'x_test.csv'), delimiter=',', skiprows=1)

# Where to output the optimal model
PICKLE_DIR = "output_models"
PICKLE_BASELINE_MODEL = 'baseline_optimal_lrmodel.pkl'
PICKLE_BASELINE_FILEPATH = os.path.join(PICKLE_DIR, PICKLE_BASELINE_MODEL)
# PICKLE_BASELINE_MODEL = 'optimal_lrmodel.pkl'
# PICKLE_BASELINE_FILEPATH = os.path.join(PICKLE_DIR, PICKLE_BASELINE_MODEL)

# Where to output the results of running on test:
RESULTS_DIR = 'output_predictions'
TEST_PRED_FILEPATH = os.path.join(RESULTS_DIR, 'yproba1_test_baseline_2.txt')

M_shape = x_tr_M784.shape
N_shape = x_va_N784.shape

N = N_shape[0]
M = M_shape[0]

y_tr_M = np.loadtxt(os.path.join(DATA_DIR, 'y_train_set.csv'), delimiter=',')
y_va_N = np.loadtxt(os.path.join(DATA_DIR, 'y_valid_set.csv'), delimiter=',')


def hyperparameterSelection(x_tr_M784, y_tr_M, x_va_N784, y_va_N, max_iter=1000):
    """ Iterate over our C_grid to see which hyperparameter gives us the best performance
    """
    C_grid = np.logspace(-9, 6, 31)
    # NOTE: This was the optimal C-value previously; saving for posterity
    # C_grid = np.asarray([0.31622776601683794])
    solver = 'lbfgs'

    lr_models_c = list()

    error_tr_lr_c = list()
    error_va_lr_c = list()
    bce_tr_lr_i = list()
    bce_va_lr_i = list()

    for C in C_grid:
        # Build and evaluate model for this C value
        lr_c = sklearn.linear_model.LogisticRegression(max_iter=max_iter, C=C, solver=solver, random_state=RANDOM_STATE)
        lr_c.fit(x_tr_M784, y_tr_M)  # Part b
        lr_models_c.append(lr_c)

        # Record training and validation set error rate
        yproba1_tr_M = lr_c.predict_proba(x_tr_M784)[:, 1]  # The probability of predicting class 1 on the training set
        yproba1_va_N = lr_c.predict_proba(x_va_N784)[:, 1]  # The probability of predicting class 1 on the validation set

        error_tr_lr_c.append(sklearn.metrics.zero_one_loss(y_tr_M, yproba1_tr_M >= 0.5))
        error_va_lr_c.append(sklearn.metrics.zero_one_loss(y_va_N, yproba1_va_N >= 0.5))

        bce_tr_lr_i.append(calc_mean_binary_cross_entropy_from_probas(y_tr_M, yproba1_tr_M))
        bce_va_lr_i.append(calc_mean_binary_cross_entropy_from_probas(y_va_N, yproba1_va_N))

    plot_error_over_c(C_grid, error_tr_lr_c, error_va_lr_c, 'output-figures/ErrorRateCValues_baseline.png')
    plot_loss_over_c(C_grid, bce_tr_lr_i, bce_va_lr_i, 'output-figures/LogLossCValues_baseline.png')

    optimal_c_index = np.argmin(error_va_lr_c)
    optimal_c = C_grid[optimal_c_index]
    print("optimal_c", optimal_c)

    # Pickle and save optimal model
    chosen_model = lr_models_c[optimal_c_index]
    with Path(PICKLE_BASELINE_FILEPATH).open('wb') as pickle_file:
        pickle.dump(chosen_model, pickle_file)
    return (optimal_c, chosen_model)


def plot_loss_over_iters(iterations, iter_step, bce_tr_lr_i, bce_va_lr_i, filepath):
    """ Helper plotter for log-loss over iteration counts
    """
    fig, ax = plt.subplots()

    # Set up the Log Loss subplot
    ax.plot(np.arange(1, iterations + 1, iter_step), bce_tr_lr_i, 'b.-', label='train')
    ax.plot(np.arange(1, iterations + 1, iter_step), bce_va_lr_i, 'r.-', label='valid')
    ax.set_title('Log Loss over iterations')
    ax.set_ylabel('loss')
    ax.set_xlabel("number of iterations")
    ax.legend()
    # plt.savefig('output-figures/LogLossIterations.png')
    plt.savefig(filepath)
    plt.show()


def plot_error_over_iters(iterations, iter_step, error_tr_lr_i, error_va_lr_i, filepath):
    """ Helper plotter for error over iteration counts
    """
    fig, ax = plt.subplots()

    # Set up the Error rate subplot
    ax.plot(np.arange(1, iterations + 1, iter_step), error_tr_lr_i, 'b:', label='train')
    ax.plot(np.arange(1, iterations + 1, iter_step), error_va_lr_i, 'r:', label='valid')
    ax.set_title('Error Rate over iterations')
    ax.set_ylabel('error')
    ax.set_xlabel("number of iterations")
    ax.legend()
    # plt.savefig('output-figures/ErrorRateIterations.png')
    plt.savefig(filepath)
    plt.show()


def lossAssessment(x_tr_M784, y_tr_M, x_va_N784, y_va_N, C=1e6):
    solver = 'lbfgs'
    iterations = 40
    iter_step = 1

    lr_models_i = list()

    error_tr_lr_i = list()
    error_va_lr_i = list()

    bce_tr_lr_i = list()
    bce_va_lr_i = list()

    for i in range(0, iterations, iter_step):
        lr_i = sklearn.linear_model.LogisticRegression(max_iter=i+1, C=C, solver=solver, random_state=RANDOM_STATE)
        lr_i.fit(x_tr_M784, y_tr_M)  # Part b
        lr_models_i.append(lr_i)

        yproba1_tr_M = lr_i.predict_proba(x_tr_M784)[:, 1]  # The probability of predicting class 1 on the training set
        yproba1_va_N = lr_i.predict_proba(x_va_N784)[:, 1]  # The probability of predicting class 1 on the validation set

        error_tr_lr_i.append(sklearn.metrics.zero_one_loss(y_tr_M, yproba1_tr_M >= 0.5))
        error_va_lr_i.append(sklearn.metrics.zero_one_loss(y_va_N, yproba1_va_N >= 0.5))

        bce_tr_lr_i.append(calc_mean_binary_cross_entropy_from_probas(y_tr_M, yproba1_tr_M))
        bce_va_lr_i.append(calc_mean_binary_cross_entropy_from_probas(y_va_N, yproba1_va_N))
    # Plot error & losses

    plot_loss_over_iters(iterations, iter_step, bce_tr_lr_i, bce_va_lr_i, 'output-figures/LogLossIterations_baseline.png')
    plot_error_over_iters(iterations, iter_step, error_tr_lr_i, error_va_lr_i, 'output-figures/ErrorRateIterations.png_baseline.png')

    # Plot ROC curve
    plot_roc_curve(y_va_N, yproba1_va_N, 'output-figures/ROC_curve_baseline_iteration_loss.png')

    # Find C value with best bce & error rates
    best_error_ind = error_va_lr_i.index(min(error_va_lr_i))
    best_bce_ind = bce_va_lr_i.index(min(bce_va_lr_i))
    print(f"number of iterations with lowest error = {best_error_ind + 1}")
    print(f"number of iterations with lowest bce = {best_bce_ind + 1}")
    # Use the best BCE index as our optimal value
    optimal_iter = best_bce_ind + 1
    chosen_model = lr_models_i[best_bce_ind]
    return (optimal_iter, chosen_model)


if __name__ == '__main__':
    # # Uncomment these two lines if you need to generate a fresh train-validation
    # # split_into_train_and_valid(n_valid_samples=2000, random_state=RANDOM_STATE)

    # # Optimize the model's hyperparameters
    # (optimal_iters, _) = lossAssessment(x_tr_M784, y_tr_M, x_va_N784, y_va_N)
    # (_, optimal_model) = hyperparameterSelection(x_tr_M784, y_tr_M, x_va_N784, y_va_N, max_iter=optimal_iters)

    # # # See the results of our optimization
    # yproba1_tr_M = optimal_model.predict_proba(x_tr_M784)[:, 1]
    # yproba1_va_N = optimal_model.predict_proba(x_va_N784)[:, 1]
    # plot_roc_curve(y_va_N, yproba1_va_N, 'output-figures/ROC_curve_baseline.png')

    # print(sklearn.metrics.confusion_matrix(y_va_N, optimal_model.predict(x_va_N784)))
    # # # Analyze the Mistakes:
    analyzeMistakes(x_va_N784, y_va_N, PICKLE_BASELINE_FILEPATH, os.path.join('output-figures', 'mistakes_baseline.png'))

    # # Interpret weights
    # # TODO:

    # Run and save results against test
    # runModelOnTest(PICKLE_BASELINE_FILEPATH, x_te_N784, TEST_PRED_FILEPATH)
    # print("Finished training, then selecting optimal baseline model and plotting related figures")
