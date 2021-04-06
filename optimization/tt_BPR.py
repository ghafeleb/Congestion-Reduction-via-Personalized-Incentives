from __future__ import print_function
import data_loader
from base import *
import time
import random
import math
import networkx as nx
import joblib
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from copy import deepcopy
import psutil
import yaml
import argparse


try:
    from collections import OrderedDict
except ImportError:
    OrderedDict = dict


def check_exists(add_, create_=True):
    if not os.path.exists(add_):
        print('Folder does not exist: \n' + add_)
        if create_:
            print('Creating the folder...')
            os.mkdir(add_)


def bpr_f_speed(volume, c_0, w_capacity):
    # c_0 = Learned free speed
    # s = Speed
    s = c_0 * (1 / (1 + 0.15*(volume/w_capacity)**4))
    return s


def EF(v_speed):
    # ####################### ******** Make sure the speed unit is km/h *************
    # Petrol, All engine sizes, Emission standard: Euro IV,
    EF_value = (523.7 - v_speed*1654.4*(10**(-2)) + (v_speed**2)*(2635.4*(10**(-4))) - (v_speed**3)*(1771.5*(10**(-6)))\
               + (v_speed**4)*(442.9*(10**(-8))))
    return EF_value


def mph2kph(s_0_mph):
    s_0_kph = s_0_mph * 1.60934
    return s_0_kph

def m2km(mile_):
    km_ = [mile_[iter] * 1.60934 for iter in range(len(mile_))]
    return km_


def get_tt_BPR_link(num_link, length_km, link_c_0_dict):
    BPR_tt = np.zeros((num_link))
    for iter_link in range(num_link):
        BPR_tt[iter_link] = length_km[iter_link] * link_c_0_dict[iter_link]
    return BPR_tt


def save_tt_BPR_link(num_link, length_km, link_c_0_dict, tt_BPR_link_add):
    tt_BPR_link = get_tt_BPR_link(num_link, length_km, link_c_0_dict) * 60
    pickle.dump(tt_BPR_link, open(tt_BPR_link_add, "wb"))


# #############################################
def get_tt_BPR_path(path_list, OD_paths, tt_BPR_link_dict):
    # vector of number of paths between each OD pair
    # num_path_v = [len(x) for x in OD_paths.itervalues()]
    num_path_v = [len(x) for x in OD_paths.values()]
    # Total number of paths
    # num_path = np.sum(num_path_v)
    max_num_path = max(num_path_v)
    OD_list = list(OD_paths.keys())
    num_OD = len(OD_list)

    # for h in xrange(N):
    #     #         print h, N
    #     start_time = (datetime.datetime.combine(tmp_date, analysis_start_time) + h * time_interval).time()
    # First finding the probabilities and assigning them ti "p" attribute of the Path objects
    for (O, D), paths in OD_paths.items():
        # print((O,D))
        for path in paths:
            #  cost is initialized so if we have no links in the path, cost is zero
            cost = 0
            for link_temp in path.link_list:
                cost += tt_BPR_link_dict[link_temp.ID]
            # "cost" is an attribute of Path objects >> Travel time of the path
            path.cost = cost

    tt_BPR_path = np.zeros((max_num_path, num_OD))
    # Second, assigning the generate probabilities to the route choice matrix
    # OD pair rs
    for rs, (O, D) in enumerate(OD_list):
        # print("\n\n\nrs:", rs)
        counter = 0
        # Path k of OD
        for k, path in enumerate(path_list):
            # Iterating over all the paths between (O, D)
            if k < np.sum(num_path_v[0:rs + 1]) and k >= np.sum(num_path_v[0:rs]):
                # print("k:", k)
                # print("counter:", counter)
                x_loc = rs
                # print("x_loc:", x_loc)
                y_loc = counter
                # print("y_loc:", y_loc)
                # Vector of probabilities of paths between OD
                data = path.cost
                tt_BPR_path[y_loc, x_loc] = data
                counter += 1
            else:
                counter = 0
    return tt_BPR_path


def save_tt_BPR_path(path_list, OD_paths, tt_BPR_link_dict, tt_BPR_path_add):
    tt_BPR_path = get_tt_BPR_path(path_list, OD_paths, tt_BPR_link_dict)
    pickle.dump(tt_BPR_path, open(tt_BPR_path_add, "wb"))


def main(args):
    # Idea: Find the free travel time of links by BPR function (or any other function), then find the travel time of paths based on their links
    # config_filename = 'YAML/region_x4_modified_tt.yaml'
    # config_filename = 'YAML/region_y1_tt.yaml'
    with open(args.config_filename, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    # save_model_ = config_dict['save_model_']
    # start_hour_opt = config_dict['start_hour_opt']
    # start_minute_opt = config_dict['start_minute_opt']
    # n_hr_test_horizon = config_dict['n_hr_test_horizon']
    region_ = config_dict['region_']
    # pad_str = config_dict['pad_str']
    # n_hours = config_dict['n_hours']
    # start_hour = config_dict['start_hour']
    # interval_t = config_dict['interval_t']
    # analysis_month = config_dict['analysis_month']
    # analysis_day = config_dict['analysis_day']
    num_paths_temp = config_dict['num_paths_temp']
    # incentive_list_available = config_dict['incentive_list_available']
    # incentive_list_temp = config_dict['incentive_list_temp']
    # new_file_temp = config_dict['new_file_temp']
    # folder_DPFE = config_dict['folder_DPFE']


    col_width = max(len(word) for word in config_dict.keys()) + 4  # padding
    for a, b in config_dict.items():
        print("".join([a.ljust(col_width), str(b)]))


    # # Initialization
    # # HOUR of start. For instance 3 to 10 for 3AM-10AM data
    # start_hour_opt = 6
    # # MINUTE of start. For instance """0 to 11""" for 3AM-10AM data >> if start_hour_opt=4 and start_minute_opt = 2: start time is 4:10 AM
    # start_minute_opt = 0
    # # Number of 5-minute time intervals of the analysis
    # n_hr_test_horizon = 1
    #
    # ub_multiplier_list = [1.5, 1.25, 1]
    #
    # ind_data = [5, 1]  # !!!!!!!!!!!!!!!!!!!!!!!!!
    # # region x2, sub1
    # # 1, 1: 3AM-10AM: padding: 3 AM to 5 AM artificial data, gradually increasing from zero, region x2 sub1, theta_opt=1, theta_OD_Estimation=1
    # # 1, 2: 3AM-10AM: padding: 3 AM to 5 AM artificial data, gradually increasing from zero, region x2 sub1, theta_opt=0.1, theta_OD_Estimation=0.1
    # # 1, 3: 3AM-10AM: padding: 3 AM to 5 AM artificial data, gradually increasing from zero, region x2 sub1, theta_opt=0.01, theta_OD_Estimation=0.01
    # # region x2, sub2
    # # 2, 1: 3AM-10AM >> padding: 3 AM to 5 AM artificial data, gradually increasing from zero, region x2 sub2, theta_opt=1, theta_OD_Estimation=1
    # # 2, 2: 3AM-10AM >> padding: 3 AM to 5 AM artificial data, gradually increasing from zero, region x2 sub2, theta_opt=0.1, theta_OD_Estimation=0.1
    # # 2, 3: 3AM-10AM >> padding: 3 AM to 5 AM artificial data, gradually increasing from zero, region x2 sub2, theta_opt=0.01, theta_OD_Estimation=0.01
    # if ind_data[0] == 1:  #
    #     region_ = 'region_x1_sub1'
    #     i_str_available = ''
    #     start_hour = 3  # Time of the start of the interval
    #     n_hours = 7  # Number of included hours
    #     interval_t = 5
    #     num_paths_temp = 5
    #     incentive_list_temp = [2, 5, 10, 20, 100]
    #     if ind_data[1] == 1:
    #         new_file_temp = '?'  # !! Starting time = 5 AM >> 12 PM # !!
    #     elif ind_data[1] == 2:
    #         new_file_temp = '?'  # !! Starting time = 5 AM >> 12 PM # !!
    #     elif ind_data[1] == 3:
    #         new_file_temp = '?'  # !! Starting time = 5 AM >> 12 PM # !!
    #
    # elif ind_data[0] == 2:  #
    #     region_ = 'region_x1_sub2'
    #     # i_str_available = ''
    #     n_hours = 7  # Number of included hours
    #     start_hour = 3  # Time of the start of the interval
    #     interval_t = 5
    #     num_paths_temp = 2
    #     incentive_list_temp = [1, 10]
    #     if ind_data[1] == 1:
    #         new_file_temp = '?'  # !! Starting time = 5 AM >> 12 PM # !!
    #     elif ind_data[1] == 2:
    #         new_file_temp = '?'  # !! Starting time = 5 AM >> 12 PM # !!
    #     elif ind_data[1] == 3:
    #         new_file_temp = '?'  # !! Starting time = 5 AM >> 12 PM # !!
    #
    # elif ind_data[0] == 3:  #
    #     # i_str_available = ''
    #     region_ = 'region_x4_modified'
    #     pad_str = '_pad'
    #     n_hours = 7  # Number of included hours
    #     start_hour = 3  # Time of the start of the interval
    #     interval_t = 15
    #     num_paths_temp = 6
    #     incentive_list_temp = [2, 5, 10, 20, 100]
    #     if ind_data[1] == 1:
    #         new_file_temp = '?'  # !! Starting time = 5 AM >> 12 PM # !!
    #     elif ind_data[1] == 2:
    #         new_file_temp = '?'  # !! Starting time = 5 AM >> 12 PM # !!
    #     elif ind_data[1] == 3:
    #         new_file_temp = '2020_09_06_11_49_58'  # !! Starting time = 5 AM >> 12 PM # !!
    #     elif ind_data[1] == 4:
    #         new_file_temp = '2020_09_07_02_29_19'  # !! Starting time = 5 AM >> 12 PM # !!
    #
    # elif ind_data[0] == 4:  #
    #     region_ = 'region_x4_modified'
    #     pad_str = '_pad'
    #     n_hours = 7  # Number of included hours
    #     start_hour = 3  # Time of the start of the interval
    #     interval_t = 15
    #     num_paths_temp = 6
    #     incentive_list_available = [1, 10, 100, 1000, 10000]
    #     # i_str_available = '_1_10_100_1000_10000'
    #     incentive_list_temp = [1, 10, 100, 1000]
    #     # i_str_temp = '_1_10_100_1000_'
    #     if ind_data[1] == 0:
    #         new_file_temp = '2020_09_15_01_40_43'
    #     elif ind_data[1] == 1:
    #         new_file_temp = '2020_09_14_14_31_08'  # !! Starting time = 5 AM >> 12 PM # !!
    #     elif ind_data[1] == 2:
    #         new_file_temp = '2020_09_13_18_46_47'  # !! Starting time = 5 AM >> 12 PM # !!
    #     elif ind_data[1] == 3:
    #         new_file_temp = '2020_09_12_19_46_15'  # !! Starting time = 5 AM >> 12 PM # !!
    #     elif ind_data[1] == 4:
    #         new_file_temp = '2020_09_12_15_03_43'  # !! Starting time = 5 AM >> 12 PM # !!
    #
    # elif ind_data[0] == 5:  #
    #     region_ = 'region_x4_modified'
    #     pad_str = '_pad'
    #     n_hours = 7  # Number of included hours
    #     start_hour = 3  # Time of the start of the interval
    #     interval_t = 15
    #     num_paths_temp = 6
    #     incentive_list_available = [1, 2, 5, 10, 1000]
    #     # i_str_available = '_' + '_'.join(str(x) for x in incentive_list_available) + '_'
    #     incentive_list_temp = [1, 2, 5, 10, 1000]
    #     # i_str_temp = '_' + '_'.join(str(x) for x in incentive_list_temp) + '_'
    #     if ind_data[1] == 0:
    #         new_file_temp = '2020_09_15_01_43_41'
    #     elif ind_data[1] == 1:
    #         new_file_temp = '2020_09_15_01_40_43'  # !! Starting time = 5 AM >> 12 PM # !!
    #     elif ind_data[1] == 2:
    #         new_file_temp = '?'  # !! Starting time = 5 AM >> 12 PM # !!
    #     elif ind_data[1] == 3:
    #         new_file_temp = '?'  # !! Starting time = 5 AM >> 12 PM # !!
    #     elif ind_data[1] == 4:
    #         new_file_temp = '?'  # !! Starting time = 5 AM >> 12 PM # !!
    #
    # elif ind_data[0] == 6:  #
    #     region_ = 'region_x4_modified2'
    #     pad_str = '_pad'
    #     n_hours = 7  # Number of included hours
    #     start_hour = 3  # Time of the start of the interval
    #     interval_t = 15
    #     num_paths_temp = 6
    #     incentive_list_available = [1, 10, 100, 1000, 10000]
    #     # i_str_available = '_1_10_100_1000_10000'
    #     incentive_list_temp = [1, 10, 100]
    #     # i_str_temp = '_1_10_100_'
    #
    #     if ind_data[1] == 1:
    #         new_file_temp = '?'  # !! Starting time = 5 AM >> 12 PM # !!
    #     elif ind_data[1] == 2:
    #         new_file_temp = '?'  # !! Starting time = 5 AM >> 12 PM # !!
    #     elif ind_data[1] == 3:
    #         new_file_temp = '?'  # !! Starting time = 5 AM >> 12 PM # !!
    #     elif ind_data[1] == 4:
    #         new_file_temp = '?'  # !! Starting time = 5 AM >> 12 PM # !!


    # finish_hour_opt = start_hour_opt + n_hr_test_horizon
    # Check day
    # analysis_date = datetime.date(random.randint(2018, 2018), random.randint(analysis_month, analysis_month), random.randint(analysis_day, analysis_day))  # !!
    # i_str_available = '_' + '_'.join(str(x) for x in incentive_list_available) + '_'
    # i_str_temp = '_' + '_'.join(str(x) for x in incentive_list_temp) + '_'
    # finish_hour = start_hour + n_hours
    # n_times_per_hr = int(60 / interval_t)
    # h_start_test_temp = n_times_per_hr * (start_hour_opt - start_hour) + math.floor(start_minute_opt/interval_t)
    # num_interval_temp = int(n_times_per_hr * n_hours)  # 7 hours, 12 5-minute intervals in each hour >> data 10AM-5PM
    # analysis_start_time_temp = datetime.time(start_hour, 0, 0)  # !! Starting time = 3 AM # !!
    # time_interval_temp = datetime.timedelta(minutes=interval_t)  # !! Time interval = 5 minutes


    # if ind_data[1] == 0:
    #     theta_temp_str = '10'
    # elif ind_data[1] == 1:
    #     theta_temp_str = '1'
    # elif ind_data[1] == 2:
    #     theta_temp_str = '1e-1'
    # elif ind_data[1] == 3:
    #     theta_temp_str = '1e-2'
    # elif ind_data[1] == 4:
    #     theta_temp_str = '1e-3'
    # folder_DPFE = 'DPFE_files'  # !!
    # OD_est_file_add_temp = folder_DPFE + '/Q_vector/' + new_file_temp + '/python3/2018-05-01.pickle'  # 10AM-5PM
    od_list_address = 'data/graph/' + region_ + '/my_od_list_' + region_ + '_original.pickle'
    graph_address = 'data/graph/' + region_ + '/my_graph_' + region_ + '_original.gpickle'
    # pck_address_ub_temp = 'data/capacity/' + region_ + '/Mar2May_2018_new_5-22_link_capacity_' + region_ + '.pickle'  # !!
    # pck_address_s_0_temp = 'data/capacity/' + region_ + '/Mar2May_2018_new_5-22_link_s_0_' + region_ + '.pickle'  # !!
    # plt_opt_address = 'plot/opt/' + region_ + i_str_available
    # check_exists(add_=plt_opt_address, create_=True)
    # q_plot_file = 'plot/q_time/' + region_ + i_str_available
    # check_exists(add_=q_plot_file, create_=True)
    # v_cap_plot_file = 'plot/VolumeVS.Capacity/' + region_
    # check_exists(add_=v_cap_plot_file, create_=True)
    # speed_s_0_plot_file = 'plot/SpeedVS.S_0/' + region_
    # check_exists(add_=speed_s_0_plot_file, create_=True)
    # random_plot_file = 'plot/random/' + region_
    # check_exists(add_=random_plot_file, create_=True)
    # tt_path_OD_plot_file = 'plot/random/' + region_ + '/tt_path_OD'
    # check_exists(add_=tt_path_OD_plot_file, create_=True)
    # tt_delta1_plot_file = 'plot/random/' + region_ + '/tt_delta1'
    # check_exists(add_=tt_delta1_plot_file, create_=True)
    # avg_tt_delta_plot_file = 'plot/random/' + region_ + '/avg_tt_delta'
    # check_exists(add_=avg_tt_delta_plot_file, create_=True)
    mile_address = 'data/graph/' + region_ + '/link_length_mile_' + region_ + '_original.pickle'
    check_exists(add_=mile_address, create_=True)

    # Loading the free flow speed
    pck_address = 'data/capacity/' + region_ + '/Mar2May_2018_new_5-22_link_s_0_' + region_ + '.pickle'
    with open(pck_address, 'rb') as f:
        link_c_0_dict0 = pickle.load(f)
    link_c_0_dict = dict()
    for i, (j, k) in enumerate(link_c_0_dict0.items()):
        link_c_0_dict[i] = mph2kph(k)
        # print(i, link_c_0_dict[i])

    # # Loading the capacity of links
    # pck_address = 'data/capacity/' + region_ + '/Mar2May_2018_new_5-22_link_capacity_' + region_ + '.pickle'
    # with open(pck_address, 'rb') as f:
    #     link_cap_dict = pickle.load(f)



    # Load list of OD pairs
    with open(od_list_address, 'rb') as handle:  # !!
        (O_list, D_list) = pickle.load(handle)
    # Load graph
    G = nx.read_gpickle(graph_address)  # !!
    G = nx.freeze(G)

    G2 = nx.read_gpickle(graph_address)  # !!

    # # Plot the graph
    # pos_ = nx.spring_layout(G)
    # nx.draw(G, pos=pos_)
    # nx.draw_networkx_edge_labels(G, pos=pos_)
    # nx.draw_networkx_labels(G, pos=pos_)
    # plt.show()

    # # Debug the path difference
    # pd.DataFrame(num_path_v).to_csv('test_num_path_py3.csv')
    # py3 = np.array(num_path_v)
    # add_py2 = 'test_num_path_py2.csv'
    # py2 = pd.read_csv(add_py2, index_col=0)
    # py2 = np.array(py2['0'].values)
    # diff_idx = (py3 - py2) != 0
    # loc_tot = np.arange(len(diff_idx))
    # loc_not = loc_tot[diff_idx]
    # py2_dist = G.adj[10136][10141]['length'] + G.adj[10141][130736]['length'] + G.adj[130736][10200]['length'] + \
    # G.adj[10200][12432]['length'] + G.adj[12432][10345]['length'] + G.adj[10345][10353]['length'] + \
    # G.adj[10353][11625]['length'] + G.adj[11625][10528]['length'] + G.adj[10528][130731]['length']
    # py3_dist = G.adj[10136][10141]['length'] + G.adj[10141][130736]['length'] + G.adj[130736][10200]['length'] + \
    # G.adj[10200][12432]['length'] + G.adj[12432][10345]['length'] + G.adj[10345][10353]['length'] + \
    # G.adj[10353][11625]['length'] + G.adj[11625][130994]['length'] + G.adj[130994][130731]['length']

    # counter = 0
    # ####oaidsfidsajf
    start_time = time.time()
    OD_paths = OrderedDict()
    # Information of all links that are traversed, set of all link classes, keys are link ids
    link_dict = OrderedDict()
    # Set of all of the Path objects, one object for each path between OD pairs
    # OD_paths also include all the paths but with specifying the OD pair of the path but in path_list all path objects are together
    path_list = list()
    for O in O_list:
        for D in D_list:
            # if not diff_idx[counter]:
            #     counter += 1
            #     continue

            # paths = list(k_shortest_paths(G, O, D, num_paths_temp)) # !! My code
            # print "From ", O, " To ", D, "there is/are ", len(paths), "path(s)"

            G_temp = deepcopy(G2)
            OD_temp = [O, D]
            paths = list()
            # path_temp = list(k_shortest_paths(G_temp, O, D, 1)) # !! My code
            # paths.append(path_temp[0])
            # if len(path_temp[0]) > 2:
            for iter_path in range(num_paths_temp):
                try:
                    path_temp = list(k_shortest_paths(G_temp, O, D, 1))[0]  # !! My code
                    # length_path = 0
                    # for iter_edge in range(len(path_temp) - 1):
                    #     length_path += \
                    #     G_temp.adj[path_temp[iter_edge]][path_temp[iter_edge + 1]]['length']
                    # print('length_path:', length_path)
                    paths.append(path_temp)
                    if len(path_temp) <= 2:
                        break
                    bool_idx = [path_temp[iter] not in OD_temp for iter in range(len(path_temp))]
                    l_remove_nodes = [path_temp[iter] for iter in range(len(path_temp)) if bool_idx[iter]==True]
                    # if the list is not empty
                    if l_remove_nodes:
                        for iter_remove in range(len(l_remove_nodes)):
                            G_temp.remove_node(l_remove_nodes[iter_remove])
                except nx.NetworkXNoPath:
                    # print('No more path between ' + str(O) + ' and ' + str(D))
                    break
            # print('paths:', paths)
            # print("From ", O, " To ", D, "there is/are ", len(paths), "path(s)\n\n")
            # If the number of paths between O and D is at least 1
            if len(paths) != 0:
                # We create tmp_path_list and fill it with the path objects in the 'base.py' code
                # Next we add all these path objects for O and D to OD_paths[(O, D)]
                tmp_path_list = list()
                for path in paths:
                    # path_o is a Path object
                    path_o = Path()
                    # this path is now an attribute of Path object
                    path_o.node_list = path
                    # Constructs the
                    path_o.node_to_list(G, link_dict)
                    tmp_path_list.append(path_o)
                    path_list.append(path_o)
                # Add the list of all Path classes for the pair O & D, Each Path class includes all the attributes
                OD_paths[(O, D)] = tmp_path_list
            # counter += 1
    print((time.time() - start_time) / 60)

    # OD_paths = OrderedDict()
    # # Information of all links that are traversed, set of all link classes, keys are link ids
    # link_dict = OrderedDict()
    # # Set of all of the Path objects, one object for each path between OD pairs
    # # OD_paths also include all the paths but with specifying the OD pair of the path but in path_list all path objects are together
    # path_list = list()
    #
    # for O in O_list:
    #     for D in D_list:
    #         # Paths from O to D
    #         # Choose the number of paths # !! My code
    #         try:
    #             paths = list(k_shortest_paths(G, O, D, num_paths_temp))  # !! My code
    #             print(paths)
    #             # print("From", O, "To ", D, "there are", len(paths), "paths")
    #             # If the number of paths between O and D is at least 1
    #             if len(paths) != 0:
    #                 # We create tmp_path_list and fill it with the path objects in the 'base.py' code
    #                 # Next we add all these path objects for O and D to OD_paths[(O, D)]
    #                 tmp_path_list = list()
    #                 for path in paths:
    #                     # path_o is a Path object
    #                     path_o = Path()
    #                     # this path is now an attribute of Path object
    #                     path_o.node_list = path
    #                     # Constructs the
    #                     path_o.node_to_list(G, link_dict)
    #
    #                     tmp_path_list.append(path_o)
    #                     path_list.append(path_o)
    #                 # Add the list of all Path classes for the pair O & D, Each Path class includes all the attributes
    #                 OD_paths[(O, D)] = tmp_path_list
    #         except nx.NetworkXNoPath:
    #             print("From", O, "To ", D, "there are NO paths")
    #             paths = list(k_shortest_paths(G, O, O, num_paths_temp))  # !! My code
    #             # If the number of paths between O and D is at least 1
    #             if len(paths) != 0:
    #                 # We create tmp_path_list and fill it with the path objects in the 'base.py' code
    #                 # Next we add all these path objects for O and D to OD_paths[(O, D)]
    #                 tmp_path_list = list()
    #                 for path in paths:
    #                     # path_o is a Path object
    #                     path_o = Path()
    #                     # this path is now an attribute of Path object
    #                     path_o.node_list = path
    #                     # Constructs the
    #                     path_o.node_to_list(G, link_dict)
    #
    #                     tmp_path_list.append(path_o)
    #                     path_list.append(path_o)
    #                 # Add the list of all Path classes for the pair O & D, Each Path class includes all the attributes
    #                 OD_paths[(O, D)] = tmp_path_list


    ## Generate Delta
    # Number of OD pairs
    num_OD = len(OD_paths)
    link_list = list(link_dict.values())
    # Sample of first 3 elements of link_dict:
    # OrderedDict([(25, < base.Link at 0x1d512b199b0 >),
    #              (55, < base.Link at 0x1d512b19a90 >),
    #              (53, < base.Link at 0x1d512b19cc0 >), ...
    # Sample of fist 3 elements of link_key_list:
    # [25,
    #  55,
    #  53, ...
    link_key_list = list(link_dict.keys())
    num_link = len(link_list)

    # .itervalues(): returns an iterator over the values of dictionary dictionary
    # vector of number of paths between each OD pair
    num_path_v = [len(x) for x in OD_paths.values()]
    # Total number of paths
    num_path = np.sum(num_path_v)
    max_num_path = max(num_path_v)
    # Number of intervals
    assert (len(path_list) == num_path)

    with open(mile_address, 'rb') as f:
        df_length_mile = pickle.load(f)
    length_mile = list(df_length_mile['length_mile'].values)
    length_km = m2km(length_mile)

    start_time = time.time()
    # # Save travel time of links based on the BPR function
    # tt_BPR_link_add = 'data/capacity/' + region_ + '/Mar2May_2018_new_5-22_link_tt_minutes_' + region_ + '.pickle'
    # save_tt_BPR_link(num_link, length_km, link_c_0_dict, tt_BPR_link_add)

    # Save travel time of paths based on the BPR function
    tt_BPR_link_add = 'data/capacity/' + region_ + '/Mar2May_2018_new_5-22_link_tt_0_minutes_' + region_ + '.pickle'
    with open(tt_BPR_link_add, 'rb') as f:
        tt_BPR_link = pickle.load(f)
    tt_BPR_link_dict = dict()
    # Iterating through the pandas DataFrame including the information of links
    for index, row in tt_BPR_link.iterrows():
        tt_BPR_link_dict[int(row['link'])] = row['tt0_minutes']

    tt_BPR_path_add = 'data/capacity/' + region_ + '/Mar2May_2018_new_5-22_path_tt_minutes_' + region_ + '.pickle'
    save_tt_BPR_path(path_list, OD_paths, tt_BPR_link_dict, tt_BPR_path_add)

    t_ = (time.time() - start_time) / 60
    print("save_tt_BPR_link & path: %.2f minutes" % t_)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename',
                        default='YAML/region_z.yaml',
                        type=str,
                        help='Configuration filename for the region.')
    args = parser.parse_args()
    main(args)
