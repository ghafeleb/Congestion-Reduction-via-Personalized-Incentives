import pickle
import argparse
import os
import numpy as np
import time
import datetime

def param_loader(pickle_address):
    with open(pickle_address, 'rb') as fh:
        W = pickle.load(fh)
        return W


def param_saver(pickle_address, param):
    with open(pickle_address, "wb") as f:
        pickle.dump(param, f)


def data_loader_f(pickle_address, shuffle_, skip_header_):
    try:
        with open(pickle_address, 'r') as fh:
            # ######################################################### Train with complete metr-la
            # ######################################################### Test with downtown-la but common sensors
            my_data = pickle.load(open(pickle_address, "rb"))
            
            return my_data

    except IOError:
        X_data = open(pickle_address, 'rt')
        x = np.genfromtxt(X_data, delimiter=",", skip_header=skip_header_)

        x_row_size = np.size(x, 0)

        idx = np.arange(0, x_row_size)
        if shuffle_:
            np.random.shuffle(idx)

        pickle.dump(x, open(pickle_address, "wb"))

        return x

def data_pickle(x, pickle_address):
    pickle.dump(x, open(pickle_address, "wb"))



