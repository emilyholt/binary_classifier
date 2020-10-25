import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from shared_code import *

DATA_DIR = 'data_sneaker_vs_sandal'


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
    print(f"Test total = {x_test_shape}\n")


def split_into_train_and_valid(n_valid_samples=2000, random_state=None):

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
    print(f"Y Train set size = {y_train_set.shape}")
    print(f"X Valid set size = {x_valid_set.shape}")
    print(f"Y Valid set size = {y_valid_set.shape}")

    train_pos = np.count_nonzero(y_train_set == 1.0)
    print(f"train_pos total = {train_pos}")
    valid_pos = np.count_nonzero(y_valid_set == 1.0)
    print(f"valid_pos total = {valid_pos}")

    np.savetxt(os.path.join(DATA_DIR, 'x_train_set.csv'), x_train_set, delimiter=',', fmt='%g')
    np.savetxt(os.path.join(DATA_DIR, 'y_train_set.csv'), y_train_set, delimiter=',', fmt='%g')
    np.savetxt(os.path.join(DATA_DIR, 'x_valid_set.csv'), x_valid_set, delimiter=',', fmt='%g')
    np.savetxt(os.path.join(DATA_DIR, 'y_valid_set.csv'), y_valid_set, delimiter=',', fmt='%g')
    print("Finished writing files")


if __name__ == '__main__':
    data_exploration()
    # For consistent splits, use a random_state param
    random_state = 100
    split_into_train_and_valid(n_valid_samples=2000, random_state=random_state)
