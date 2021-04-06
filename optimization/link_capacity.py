import matplotlib.pyplot as plt
import numpy as np
import pickle
import data_loader
import address
import networkx as nx
import time
import numpy as np
import pandas as pd
import datetime
import os
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import pickle
from collections import OrderedDict
import copy
from scipy.sparse import csr_matrix
from scipy import io
import seaborn as sns
import joblib
from joblib import Parallel, delayed
import random
import math
import data_loader
from base_DPFE_deprecated import *
# from base import *
import time
from scipy.optimize import curve_fit
import torch
from torch.utils.data import DataLoader
# import link_capacity_train
import address
import yaml
import argparse

try:
    from collections import OrderedDict
except ImportError:
    OrderedDict = dict

class BPR_function(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c_0 = torch.nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.w = torch.nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

        # self.f1 = self.c_0 * torch.pow(1.0 + 0.15 * torch.pow(v/self.w, 4), -1)

    def forward(self, v):
        # y_prediction = self.f1(x)
        return self.c_0 * torch.pow(1.0 + 0.15 * torch.pow(v/self.w, 4), -1)


def main(args):
    # config_filename = 'YAML_BPR/region_x4_modified.yaml'
    # config_filename = 'YAML_BPR/region_y1.yaml'
    with open(args.config_filename, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    region_ = config_dict['region_']
    start_hr = config_dict['start_hr']
    finish_hr = config_dict['finish_hr']
    interval_t = config_dict['interval_t']
    data_year = config_dict['data_year']
    months_ = config_dict['months_']
    n_train = config_dict['n_train']


    col_width = max(len(word) for word in config_dict.keys()) + 4  # padding
    for a, b in config_dict.items():
        print("".join([a.ljust(col_width), str(b)]))
    # Load data
    # region_ = 'region_x4_modified'
    if not os.path.exists('plot'):
        print('Creating folder: \n' + 'plot')
        os.mkdir('plot')
    else:
        print('Folder exists: \n' + 'plot')

    if not os.path.exists('plot/fit_curve'):
        print('Creating folder: \n' + 'plot/fit_curve')
        os.mkdir('plot/fit_curve')
    else:
        print('Folder exists: \n' + 'plot/fit_curve')

    if not os.path.exists('plot/fit_curve/loss_function'):
        print('Creating folder: \n' + 'plot/fit_curve/loss_function')
        os.mkdir('plot/fit_curve/loss_function')
    else:
        print('Folder exists: \n' + 'plot/fit_curve/loss_function')

    if not os.path.exists('plot/fit_curve/loss_function/' + region_):
        print('Creating folder: \n' + 'plot/fit_curve/loss_function/' + region_)
        os.mkdir('plot/fit_curve/loss_function/' + region_)
    else:
        print('Folder exists: \n' + 'plot/fit_curve/loss_function/' + region_)

    if not os.path.exists('plot/fit_curve/fit_data'):
        print('Creating folder: \n' + 'plot/fit_curve/fit_data')
        os.mkdir('plot/fit_curve/fit_data')
    else:
        print('Folder exists: \n' + 'plot/fit_curve/fit_data')

    if not os.path.exists('plot/fit_curve/fit_data/' + region_):
        print('Creating folder: \n' + 'plot/fit_curve/fit_data/' + region_)
        os.mkdir('plot/fit_curve/fit_data/' + region_)
    else:
        print('Folder exists: \n' + 'plot/fit_curve/fit_data/' + region_)


    # ID of sensors in the selected region
    # ID_address = 'data/graph/my_ID_region_x2_sub1_original.pickle'
    # ID_address = 'data/graph/my_ID_region_x2_sub2_original.pickle'
    ID_address = 'data/graph/' + region_ + '/my_ID_' + region_ + '_original.pickle'
    IDs = data_loader.data_loader_f(ID_address, shuffle_=False, skip_header_=False, name_=False) # !! New pickled version

    name_temp = str(months_) + '_'+ str(data_year) +'_new_' + str(start_hr) + '-' + str(finish_hr) + '_' + region_
    speed_data = pd.read_csv('data/speed_volume/' + str(months_) + '_'+ str(data_year) +'_AVG' + str(interval_t) +
                             '_' + str(start_hr) + '-' + str(finish_hr) + '_' + region_ + '_with_linkID/speed_' +
                             name_temp + '.csv',
                             encoding='UTF-8', header=None, index_col=False)  # !!
    # Volume data
    volume_data = pd.read_csv('data/speed_volume/' + str(months_) + '_'+ str(data_year) +'_AVG' + str(interval_t) +
                              '_' + str(start_hr) + '-' + str(finish_hr) + '_' + region_ + '_with_linkID/volume_' +
                              name_temp + '.csv',
                              encoding='UTF-8', header=None, index_col=False)  # !!
    # Create ROW indecies of dataframe in Pandas
    month_dict = dict()
    # Business days of March 2018
    month_dict[3] = [1, 2, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 26, 27, 28, 29, 30]  # !!
    # Business days of April 2018
    month_dict[4] = [2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 23, 24, 25, 26, 27, 30]  # !!
    # Business days of May 2018
    month_dict[5] = [1, 2, 3, 4, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 29, 30, 31]  # !!
    # Number of days for each ID
    num_days = sum([len(x) for x in month_dict.values()]) # python 3
    date_need_to_finish = list()
    for iter_month in month_dict.keys():
        for iter_day in month_dict[iter_month]:
            # print '\n'
            date_temp = datetime.date(2018, iter_month, iter_day)  # !!
            time_basis = datetime.time(5, 0, 0)  # !! 5 AM
            cur_date_time = datetime.datetime.combine(date_temp, time_basis)
            # print cur_date_time
            single_date = cur_date_time.date()
            # print single_date
            date_need_to_finish.append(single_date)
    # Create COLUMN labels of dataframe in Pandas
    num_time = 17*12 # 17 hours, intervals = 5 minutes >> 17*12 codes # !!
    t = [datetime.time(5 + int(iter / 12), iter % 12 * 5, 00) for iter in range(num_time)]  # !! 5 AM - 10 PM, Python 3

    speed_dict = dict()
    volume_dict = dict()
    for iter_ID in range(IDs.shape[0]):
        ID_ = IDs.loc[iter_ID, 0]
        # Speed data
        temp_idx_speed = speed_data[0].isin([ID_.astype(int)])
        temp_speed_data = speed_data.loc[temp_idx_speed, 1:]
        temp_speed_data.index = date_need_to_finish
        temp_speed_data.columns = t
        speed_dict[ID_] = temp_speed_data
        # Volume data
        temp_idx_vol = volume_data[0].isin([ID_.astype(int)])
        temp_volume_data = volume_data.loc[temp_idx_vol, 1:]
        temp_volume_data.index = date_need_to_finish
        temp_volume_data.columns = t
        volume_dict[ID_] = temp_volume_data

    # ################## BPR + PyTorch ##################
    # v: volume data, c_0: speed at zero volume, w: capacity
    # inventory_add = 'data/congestion_inventory/congestion_inventory(original).csv'
    # inv_ = pd.read_csv(inventory_add, encoding='UTF-8', header=0, index_col=False)

    def bpr_f(v, c_0, w):
        s = c_0 * (1 / (1 + 0.15*(v/w) ** 4))
        return s

    gpu_ = False

    # torch.manual_seed(42)

    if gpu_:
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    if torch.cuda.is_available():
        if gpu_:
            device = 'cuda'
        else:
            device = 'cpu'


    sensor_cap_dict = dict()
    sensor_c_0_dict = dict()
    n_data = speed_dict[IDs.loc[0, 0]].size
    percentage_ = 1

    # l_ = [10091]

    start_time = time.time()
    for iter_ID in range(IDs.shape[0]):
    # for ID_ in l_:
        # if iter_ID % 10 == 0:
        min_loss = float('inf')
        counter_ = 0
        successful = False
        print('\n\n')
        while not successful:
            counter_ += 1
            ID_ = IDs.loc[iter_ID, 0]
            print('ID: ', ID_)
            y = np.array(speed_dict[ID_]).reshape(-1, )
            # idx_ = list(range(n_data))
            # np.random.shuffle(idx_)
            # idx_2 = idx_[:int(n_data*percentage_)]
            # # Add noise
            # y += np.random.rand(y.shape[0])
            x = (np.array(volume_dict[ID_]).reshape(-1, )).astype(float)

            train_tens_x = torch.Tensor(x).to(device)
            train_tens_y = torch.Tensor(y).to(device)

            # Now we can create a model and send it at once to the device
            model = BPR_function().to(device)
            # Sets model to TRAIN mode
            model.train()

            loss_fn = torch.nn.MSELoss(reduction='mean')

            # For each epoch...
            i = 1
            # Epochs = [5000, 5000, 5000, 5000, 5000, 5000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
            # step_size = [1, 10 ** (-i), 10 ** (-i - 1), 10 ** (-i - 2), 10 ** (-i - 3), 10 ** (-i - 4), 10 ** (-i - 5),
            #              10 ** (-i - 6), 10 ** (-i - 7)]

            Epochs = [1000, 1000, 500, 100, 100]
            step_size = [10 ** (-i), 10 ** (-i - 1), 10 ** (-i - 2), 10 ** (-i - 3), 10 ** (-i - 4)]

            # Epochs = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
            # step_size = [10 ** (-i - 1), 3 * 10 ** (-i - 2), 10 ** (-i - 2),
            #              3 * 10 ** (-i - 3), 10 ** (-i - 3), 3 * 10 ** (-i - 4), 10 ** (-i - 4), 3 * 10 ** (-i - 5), 10 ** (-i - 5)]
            w_regul = 0
            losses_temp = []
            print('Model parameters:', model.state_dict())

            for iter_lr in range(np.size(Epochs)):
                lr_ = step_size[iter_lr]
                # optimizer = torch.optim.SGD(model.parameters(), lr=lr_)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr_, weight_decay=w_regul)
                for iter_epoch in range(Epochs[iter_lr]):
                    # # Creates the train_step function for our model, loss function and optimizer
                    # train_step = make_train_step(model, loss_fn, optimizer)
                    # Makes predictions
                    yhat = model(train_tens_x)
                    # Computes loss
                    loss = loss_fn(train_tens_y, yhat)
                    # Computes gradients
                    loss.backward()
                    # Updates parameters and zeroes gradients
                    optimizer.step()
                    optimizer.zero_grad()

                    # Performs one train step and returns the corresponding loss
                    loss = loss.item()
                    # print(loss)
                    losses_temp.append(loss)
                print('Learning rate: ', lr_)
                print('Loss: ', loss)
                print('Model parameters:', model.state_dict())

            if counter_ > n_train:
                successful = True


            if loss < min_loss:
                min_loss = loss
                losses = losses_temp

                for name, param in model.named_parameters():
                    temp__ = param.data[0].item()
                    # Closest integer
                    if name == 'c_0':
                        sensor_c_0_dict[ID_] = abs(int(np.around(temp__)))
                        print(name, ':\t', sensor_c_0_dict[ID_])
                        if sensor_c_0_dict[ID_] == 0 and counter_ > 5:
                            successful = False
                            min_loss = float('inf')
                            counter_ = 0

                    else:
                        sensor_cap_dict[ID_] = abs(int(np.around(temp__)))
                        print(name, ':\t', sensor_cap_dict[ID_])
                        if sensor_cap_dict[ID_] == 0 and counter_ > 5:
                            successful = False
                            min_loss = float('inf')
                            counter_ = 0
                        if sensor_cap_dict[ID_]>200:
                            sensor_cap_dict[ID_] = 200

        fig1 = plt.figure()
        plt.plot(losses)
        plt.title('Loss value, Sensor ID: %i' % ID_)
        plt.xlabel('Epoch')
        plt.ylabel('Loss value')
        # plt.show()
        fig1.savefig('plot/fit_curve/loss_function/' + region_ + '/' + str(ID_) + '_loss.png')
        plt.close()

        x_sort = x[np.argsort(x)]
        y_original = y[np.argsort(x)]
        y_plot = bpr_f(x_sort, sensor_c_0_dict[ID_], sensor_cap_dict[ID_])
        fig2 = plt.figure()
        plt.plot(x_sort, y_plot, 'r-',
                 label='fit: c_0=%5.3f, w=%5.3f' % (sensor_c_0_dict[ID_], sensor_cap_dict[ID_]))
        # plt.plot(x_sort, y_original, 'b--', label='Original data')
        plt.scatter(x_sort, y_original, label='fit: c_0=%5.3f, w=%5.3f' % (sensor_c_0_dict[ID_], sensor_cap_dict[ID_]))
        plt.title('Fitting data to BPR function, Sensor ID: %i' % ID_)
        plt.xlabel('Volume')
        plt.ylabel('Speed')
        plt.legend()
        # plt.show()
        fig2.savefig('plot/fit_curve/fit_data/' + region_ + '/sensor_' + str(ID_) + '_pytorch.png')
        plt.close()

    tot_time = (time.time() - start_time)/60
    print(tot_time)

    # Save the capacity of sensors as DataFrame as CSV file
    d = {'sensor': list(sensor_cap_dict.keys()), 'capacity': list(sensor_cap_dict.values())}
    df = pd.DataFrame(data=d)
    if not os.path.exists('data/capacity/' + region_ ):
        print('Creating folder: \n' + 'data/capacity/' + region_ )
        os.mkdir('data/capacity/' + region_ )
    else:
        print('Folder exists: \n' + 'Creating folder: \n' + 'data/capacity/' + region_ )
    pd_address = 'data/capacity/' + region_ + '/Mar2May_2018_new_5-22_sensor_capacity_' + region_ + '.csv'
    df.to_csv(pd_address)
    # Save the capacity of sensors as Dictionary as pickle file
    pck_address = 'data/capacity/' + region_ + '/Mar2May_2018_new_5-22_sensor_capacity_' + region_ + '.pickle'
    data_loader.data_pickle(sensor_cap_dict, pck_address)

    # Save the 'free speed' of sensors as DataFrame as CSV file
    d = {'sensor': list(sensor_c_0_dict.keys()), 'c_0': list(sensor_c_0_dict.values())}
    df = pd.DataFrame(data=d)
    pd_address = 'data/capacity/' + region_ + '/Mar2May_2018_new_5-22_sensor_s_0_' + region_ + '.csv'
    df.to_csv(pd_address)
    # Save the capacity of sensors as Dictionary as pickle file
    pck_address = 'data/capacity/' + region_ + '/Mar2May_2018_new_5-22_sensor_s_0_' + region_ + '.pickle'
    data_loader.data_pickle(sensor_c_0_dict, pck_address)

    # #################### Capacity of links, region x?
    # Read graph data
    with open('data/graph/' + region_ + '/my_od_list_' + region_ + '_original.pickle', 'rb') as handle:  # !!
        (O_list, D_list) = pickle.load(handle)
    G = nx.read_gpickle('data/graph/' + region_ + '/my_graph_' + region_ + '_original.gpickle')  # !!
    G = nx.freeze(G)

    # Load the capacity of sensors as Dictionary as pickle file
    pck_address = 'data/capacity/' + region_ + '/Mar2May_2018_new_5-22_sensor_capacity_' + region_ + '.pickle'
    with open(pck_address, 'rb') as f:
        sensor_cap_dict = pickle.load(f)

    # Load the adjacency matrix
    adj_address0 = 'data/graph/' + region_ + '/AdjMatrix_' + region_ + '_original'
    adj_address = adj_address0 + '.csv'
    adj_matrix = pd.read_csv(adj_address, index_col=False)
    # adj_address_pickle = 'data/graph/AdjMatrix_' + region_ + '_original.pickle'
    # adj_matrix = data_loader.data_loader_f(adj_address_pickle, shuffle_=False, skip_header_=False, name_=False)
    adj_matrix_drop = adj_matrix.drop(['direction'], axis=1)

    l_edges = list(G.edges())
    link_cap_dict = dict()
    for iter_link in range(len(l_edges)):
        print('nodes:', l_edges[iter_link])
        loc_link = (adj_matrix['origin'] == l_edges[iter_link][0]).values * (adj_matrix['destination'] == l_edges[iter_link][1]).values
        print('adj matrix: ', adj_matrix.loc[loc_link, :])
        print('adj matrix drop: ', adj_matrix_drop.loc[loc_link, :])
        temp_ = np.unique(np.array(adj_matrix_drop.loc[loc_link]))
        temp_list = list(temp_[~np.isnan(temp_)].astype(int))
        print('temp_list:', temp_list)
        list_cap_temp0 = [sensor_cap_dict[i] for i in temp_list]
        print('list_cap_temp0', list_cap_temp0)
        # link_cap_temp = (sensor_cap_dict[l_edges[iter_link][0]] + sensor_cap_dict[l_edges[iter_link][1]]) / 2  # DEPRECATED
        link_cap_temp = np.mean(list_cap_temp0)
        print('link_cap_temp', link_cap_temp)
        link_cap_dict[iter_link] = int(np.around(link_cap_temp))
        print('link %i:' % iter_link, link_cap_dict[iter_link])
        print('\n\n\n')
        # break

    # Save the capacity of links as DataFrame as CSV file
    d = {'link': list(link_cap_dict.keys()), 'capacity': list(link_cap_dict.values())}
    df = pd.DataFrame(data=d)
    pd_address = 'data/capacity/' + region_ + '/Mar2May_2018_new_5-22_link_capacity_' + region_ + '.csv'
    df.to_csv(pd_address)
    # Save the capacity of links as Dictionary as pickle file
    pck_address = 'data/capacity/' + region_ + '/Mar2May_2018_new_5-22_link_capacity_' + region_ + '.pickle'
    data_loader.data_pickle(link_cap_dict, pck_address)

    # #################### Speed of links, region x?
    # Read graph data
    with open('data/graph/' + region_ + '/my_od_list_' + region_ + '_original.pickle', 'rb') as handle:  # !!
        (O_list, D_list) = pickle.load(handle)
    G = nx.read_gpickle('data/graph/' + region_ + '/my_graph_' + region_ + '_original.gpickle')  # !!
    G = nx.freeze(G)

    # Load the speed of sensors as Dictionary as pickle file
    pck_address = 'data/capacity/' + region_ + '/Mar2May_2018_new_5-22_sensor_s_0_' + region_ + '.pickle'
    with open(pck_address, 'rb') as f:
        sensor_c_0_dict = pickle.load(f)

    # Load the adjacency matrix
    adj_address0 = 'data/graph/' + region_ + '/AdjMatrix_' + region_ + '_original'
    adj_address = adj_address0 + '.csv'
    adj_matrix = pd.read_csv(adj_address, index_col=False)
    # adj_address_pickle = 'data/graph/AdjMatrix_' + region_ + '_original.pickle'
    # adj_matrix = data_loader.data_loader_f(adj_address_pickle, shuffle_=False, skip_header_=False, name_=False)
    adj_matrix_drop = adj_matrix.drop(['direction'], axis=1)

    l_edges = list(G.edges())
    link_c_0_dict = dict()
    for iter_link in range(len(l_edges)):
        print('nodes:', l_edges[iter_link])
        loc_link = (adj_matrix['origin'] == l_edges[iter_link][0]).values * (adj_matrix['destination'] == l_edges[iter_link][1]).values
        print('adj matrix: ', adj_matrix.loc[loc_link, :])
        print('adj matrix drop: ', adj_matrix_drop.loc[loc_link, :])
        temp_ = np.unique(np.array(adj_matrix_drop.loc[loc_link]))
        temp_list = list(temp_[~np.isnan(temp_)].astype(int))
        print('temp_list:', temp_list)
        list_cap_temp0 = [sensor_c_0_dict[i] for i in temp_list]
        print('list_cap_temp0', list_cap_temp0)
        # link_cap_temp = (sensor_c_0_dict[l_edges[iter_link][0]] + sensor_c_0_dict[l_edges[iter_link][1]]) / 2  # DEPRECATED
        link_cap_temp = np.mean(list_cap_temp0)
        print('link_cap_temp', link_cap_temp)
        link_c_0_dict[iter_link] = round(link_cap_temp, 2)
        print('link %i:' % iter_link, link_c_0_dict[iter_link])
        print('\n\n\n')
        # break

    # Save the speed of links as DataFrame as CSV file
    d = {'link': list(link_c_0_dict.keys()), 'speed': list(link_c_0_dict.values())}
    df = pd.DataFrame(data=d)
    pd_address = 'data/capacity/' + region_ + '/Mar2May_2018_new_5-22_link_s_0_' + region_ + '.csv'
    df.to_csv(pd_address)
    # Save the capacity of links as Dictionary as pickle file
    pck_address = 'data/capacity/' + region_ + '/Mar2May_2018_new_5-22_link_s_0_' + region_ + '.pickle'
    data_loader.data_pickle(link_c_0_dict, pck_address)


    # #################### Free travel time of links, region x?
    # Loading the free speed of the link (mph)
    pck_address = 'data/capacity/' + region_ + '/Mar2May_2018_new_5-22_link_s_0_' + region_ + '.pickle'
    with open(pck_address, 'rb') as f:
        link_c_0_dict = pickle.load(f)

    graph_folder = 'data/graph/' + region_
    print(f'graph_folder: {graph_folder}')

    # Distance of the link in miles
    mile_address = graph_folder + '/link_length_mile_' + region_ + '_original.pickle'
    with open(mile_address, 'rb') as f:
        df_length_mile = pickle.load(f)
    length_mile = list(df_length_mile['length_mile'].values)

    # Computing free travel time
    # tt0 = (length of the link)/(free speed) = length_mile/s_0_l
    s_0_l = list(link_c_0_dict.values())

    tt0_hour = [length_mile[iter]/s_0_l[iter] for iter in range(len(s_0_l))]
    # Save the free travel time of links as DataFrame as CSV file
    d_tt1 = {'link': list(link_c_0_dict.keys()), 'tt0_hours': tt0_hour}
    df_tt1 = pd.DataFrame(data=d_tt1)
    pd_address = 'data/capacity/' + region_ + '/Mar2May_2018_new_5-22_link_tt_0_hours_' + region_ + '.csv'
    df_tt1.to_csv(pd_address)
    # Save the capacity of links as Dictionary as pickle file
    pck_address = 'data/capacity/' + region_ + '/Mar2May_2018_new_5-22_link_tt_0_hours_' + region_ + '.pickle'
    data_loader.data_pickle(df_tt1, pck_address)

    tt0_minutes = [tt0_hour[iter]*60 for iter in range(len(tt0_hour))]
    # Save the free travel time of links as DataFrame as CSV file
    d_tt = {'link': list(link_c_0_dict.keys()), 'tt0_minutes': tt0_minutes}
    df_tt = pd.DataFrame(data=d_tt)
    pd_address = 'data/capacity/' + region_ + '/Mar2May_2018_new_5-22_link_tt_0_minutes_' + region_ + '.csv'
    df_tt.to_csv(pd_address)
    # Save the capacity of links as Dictionary as pickle file
    pck_address = 'data/capacity/' + region_ + '/Mar2May_2018_new_5-22_link_tt_0_minutes_' + region_ + '.pickle'
    data_loader.data_pickle(df_tt, pck_address)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename',
                        # default='YAML_BPR/region_x4_modified.yaml',
                        default='YAML_BPR/region_z.yaml',
                        type=str,
                        help='Configuration filename for the region.')
    args = parser.parse_args()
    main(args)
