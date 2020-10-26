import pickle
import os
import numpy as np
from shared_code import analyzeMistakes

# Pickling locations
PICKLE_DIR = "output_models"
PICKLE_BASELINE_MODEL = 'baseline_optimal_lrmodel.pkl'
PICKLE_BASELINE_FILEPATH = os.path.join(PICKLE_DIR, PICKLE_BASELINE_MODEL)
PICKLE_KNN_MODEL = 'optimal_knn_model.pkl'
PICKLE_KNN_FILEPATH = os.path.join(PICKLE_DIR, PICKLE_KNN_MODEL)

# Data location
DATA_DIR = 'data_sneaker_vs_sandal'
x_va_M784 = np.loadtxt(os.path.join(DATA_DIR, 'x_valid_set.csv'), delimiter=',')
y_va_M = np.loadtxt(os.path.join(DATA_DIR, 'y_valid_set.csv'), delimiter=',')

# Output locations
OUTPUT_DIR = 'output-figures'
MISTAKES_BASELINE_FILEPATH = os.path.join(OUTPUT_DIR, 'mistakes_baseline.png')
MISTAKES_KNN_FILEPATH = os.path.join(OUTPUT_DIR, 'mistakes_knn.png')


if __name__ == '__main__':
    print("Getting mistake figures for two models")
    analyzeMistakes(x_va_M784, y_va_M, PICKLE_BASELINE_FILEPATH, MISTAKES_BASELINE_FILEPATH)
    analyzeMistakes(x_va_M784, y_va_M, PICKLE_KNN_FILEPATH, MISTAKES_KNN_FILEPATH)
    print("Finished plotting mistake examples figures for two models")
