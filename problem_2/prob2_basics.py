import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sklearn.linear_model
import sklearn.metrics

def data_exploration():
    print("calling function")
    DATA_DIR = os.path.join('.', 'data_sneaker_vs_sandal') #Make sure you have downloaded data and your directory is correct

    # Load outcomes y arrays
    y_train = np.loadtxt(os.path.join(DATA_DIR, 'y_train.csv'), delimiter=',', skiprows=1)

    train_total = len(y_train)
    train_pos = np.count_nonzero(y_train == 1.0)
    train_frac_pos = float(train_pos) / float(train_total)

    print(f"Train total = {train_total}")
    print(f"train_pos total = {train_pos}")
    print(f"train_frac_pos = {train_frac_pos}")

if __name__ == '__main__':
    data_exploration()