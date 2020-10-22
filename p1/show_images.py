'''
Simple script to visualize 28x28 images stored in csv files

Usage
-----
$ python show_images.py --dataset_path data_sandal_vs_sneaker/

Expected Output
---------------
An active figure displaying 9 sample images arranged in 3x3 grid

'''

import argparse
import pandas as pd
import numpy as np
import os
import sys
from matplotlib import pyplot as plt

FALSE_POSITIVES_FILE = "false_positives_indices.txt"
FALSE_NEGATIVES_FILE = "false_negatives_indices.txt"


def labelDisplay(y_label):
    if (y_label == 0):
        return "8"
    elif (y_label == 1):
        return "9"
    else:
        print("Unexpected label value of " % y_label)
        return "Unknown"


def rowDisplay(row):
    return str(row + 1)


def colDisplay(col):
    # 65 == 'A' in ASCII
    return chr(col + 65)


def show_images(X, y, row_ids, n_rows=3, n_cols=3):
    ''' Display images

    Args
    ----
    X : 2D array, shape (N, 784)
        Each row is a flat image vector for one example
    y : 1D array, shape (N,)
        Each row is label for one example
    row_ids : list of int
        Which rows of the dataset you want to display
    '''
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols,
        figsize=(n_cols * 3, n_rows * 3))

    for ii, row_id in enumerate(row_ids):
        cur_ax = axes.flatten()[ii]
        cur_ax.imshow(X[row_id].reshape(28, 28), interpolation='nearest', vmin=0, vmax=1, cmap='gray')
        cur_ax.set_xticks([])
        cur_ax.set_yticks([])
        row = ii // n_cols
        col = ii % n_rows
        cur_ax.set_title(f'{rowDisplay(row)}{colDisplay(col)}\nTrue Label={labelDisplay(y[row_id])}')
    plt.show()


if __name__ == '__main__':
    default_path = 'data_digits_8_vs_9_noisy'   # Make sure you have downloaded data and your directory is correct
    path, _ = os.path.split(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default=default_path)
    parser.add_argument('--n_rows', type=int, default=3)
    parser.add_argument('--n_cols', type=int, default=3)
    parser.add_argument('--indices_path', default=os.path.join(default_path, FALSE_POSITIVES_FILE))
    args = parser.parse_args()

    indices_to_show = args.n_rows * args.n_cols
    error_indices_K = np.loadtxt(os.path.join(path, args.indices_path)).astype(int)
    row_ids_to_show = error_indices_K[:indices_to_show]

    dataset_path = args.dataset_path

    x_df = pd.read_csv(os.path.join(path, dataset_path, 'x_valid.csv'))
    x_NF = x_df.values

    y_df = pd.read_csv(os.path.join(path, dataset_path, 'y_valid.csv'))
    y_N = y_df.values

    show_images(x_NF, y_N, row_ids_to_show, args.n_rows, args.n_cols)
