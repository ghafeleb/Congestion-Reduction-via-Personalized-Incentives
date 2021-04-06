import pickle
import argparse
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import address


def param_loader(pickle_address):
    with open(pickle_address, 'rb') as fh:
        W = pickle.load(fh)
        return W


def param_saver(pickle_address, param):
    with open(pickle_address, "wb") as f:
        pickle.dump(param, f)


def data_loader_f(pickle_file_, shuffle_, skip_header_, name_ = True):
    try:
        if name_ == True:
            pickle_address = address.pickle_f(pickle_file_)
        else:
            pickle_address = pickle_file_
        with open(pickle_address, 'rb') as fh:
            my_data = pickle.load(fh)
            return my_data

    except IOError:
        csv_file, _ = address.not_pickle_f(pickle_file_)
        X_data = open(csv_file, 'rt')
        # x = np.genfromtxt(X_data, delimiter=",", skip_header=skip_header_)
        x = np.genfromtxt(X_data, delimiter=",")
        x_row_size = np.size(x, 0)
        idx = np.arange(0, x_row_size)
        if shuffle_:
            np.random.shuffle(idx)

        pickle_address = address.pickle_f(pickle_file_)
        pickle.dump(x, open(pickle_address, "wb"))
        return x


def data_pickle(x, pickle_address):
    pickle.dump(x, open(pickle_address, "wb"))


def data_loader_f_test(pickle_file_):
    try:
        pickle_address = address.pickle_f(pickle_file_)
        with open(pickle_address, 'rb') as fh:
            my_data = pickle.load(fh)
            test_tens_x = my_data[0]
            test_tens_z = my_data[1]
            test_tens_x = torch.Tensor(test_tens_x)

            return test_tens_x, test_tens_z

    except FileNotFoundError:
        csv_file_1, csv_file_2 = address.not_pickle_f(pickle_file_)
        pickle_address = address.pickle_f(pickle_file_)
        X_data = open(csv_file_1, 'rt')
        Z_data = open(csv_file_2, 'rt')
        x = np.genfromtxt(X_data, delimiter=",")
        z = np.genfromtxt(Z_data, delimiter=",")

        # MinMax normalization, range: [0, 70]
        max_ = 70
        min_ = 0
        for iter_ in range(12):
            # Normalizing day
            X_std = (x[:, iter_*3] - x[:, iter_*3].min(axis=0)) / (x[:, iter_*3].max(axis=0) - x[:, iter_*3].min(axis=0))
            x[:, iter_*3] = X_std * (max_ - min_) + min_
            # Normalizing time
            X_std = (x[:, iter_*3+1] - x[:, iter_*3+1].min(axis=0)) / (x[:, iter_*3+1].max(axis=0) - x[:, iter_*3+1].min(axis=0))
            x[:, iter_*3+1] = X_std * (max_ - min_) + min_
        # scaler = MinMaxScaler(feature_range=[0, 100], copy=True)
        # scaler.fit(x)
        # x = scaler.transform(x)

        test_tens_x = x
        test_tens_z = z

        # test_tens_x = x[idx, :]
        # test_tens_z = z[idx, :]

        test_tens_x = torch.Tensor(test_tens_x)

        pickle.dump([test_tens_x, test_tens_z], open(pickle_address, "wb"))

        return test_tens_x, test_tens_z