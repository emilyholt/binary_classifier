import pickle
import os
import numpy as np
from shared_code import runModelOnTest

import sklearn.linear_model
import sklearn.metrics
import sklearn.neighbors

# Pickling locations
PICKLE_DIR = "output_models"
PICKLE_BASELINE_MODEL = 'baseline_optimal_lrmodel.pkl'
PICKLE_BASELINE_FILEPATH = os.path.join(PICKLE_DIR, PICKLE_BASELINE_MODEL)
PICKLE_KNN_MODEL = 'optimal_knn_model.pkl'
PICKLE_KNN_FILEPATH = os.path.join(PICKLE_DIR, PICKLE_KNN_MODEL)

# Data location
DATA_DIR = 'data_sneaker_vs_sandal'
x_te_N784 = np.loadtxt(os.path.join(DATA_DIR, 'x_test.csv'), delimiter=',', skiprows=1)

# Output locations
RESULTS_DIR = 'output_predictions'
TEST_PRED_BASELINE_FILEPATH = os.path.join(RESULTS_DIR, 'yproba1_test_baseline.txt')
TEST_PRED_KNN_FILEPATH = os.path.join(RESULTS_DIR, 'yproba1_test_knn.txt')


if __name__ == '__main__':
    print("Getting mistake figures for two models")
    runModelOnTest(PICKLE_BASELINE_FILEPATH, x_te_N784, TEST_PRED_BASELINE_FILEPATH)
    runModelOnTest(PICKLE_KNN_FILEPATH, x_te_N784, TEST_PRED_KNN_FILEPATH)
    print("Finished plotting mistake examples figures for two models")
