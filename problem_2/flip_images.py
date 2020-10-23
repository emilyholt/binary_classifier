# Source: https://moonbooks.org/Articles/How-to-flip-an-image-vertically-in-python-/

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

import csv
import os
import sys

def show_images(X, row_id):
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
    fig, ax = plt.subplots()
    original_X = X[row_id].reshape(28,28)
    ax.imshow(original_X, interpolation='nearest', vmin=0, vmax=1, cmap='gray')
    # plt.show()
    flipped_x = np.fliplr(original_X)
    ax.imshow(flipped_x, interpolation='nearest', vmin=0, vmax=1, cmap='gray')
    plt.show()

# Load csv data
dataset_path = './data_sneaker_vs_sandal'
x_df = pd.read_csv(os.path.join(dataset_path, 'x_train.csv'))
x_NF = x_df.values

with open(os.path.join(dataset_path, 'x_train_augmented.csv'), 'a') as output:
	output_writer = csv.writer(output)
	for x in x_NF:
		# print(f"x = {x}")
		original_X = x.reshape(28,28)
		flipped_x = np.fliplr(original_X)
		# print(f"flipped_x = {flipped_x.flatten()}")
		output_writer.writerow(flipped_x.flatten())


