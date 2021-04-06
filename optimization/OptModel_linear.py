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
    EF_value = 523.7 - v_speed*1654.4*(10**(-2)) + (v_speed**2)*(2635.4*(10**(-4))) - (v_speed**3)*(1771.5*(10**(-6)))\
               + (v_speed**4)*(442.9*(10**(-8)))
    return EF_value


def mph2kph(s_0_mph):
    s_0_kph = s_0_mph * 1.60934
    return s_0_kph

def m2km(mile_):
    km_ = [mile_[iter] * 1.60934 for iter in range(len(mile_))]
    return km_


def main(args):
    with open(args.config_filename, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    save_model_ = config_dict['save_model_']
    start_hour_opt = config_dict['start_hour_opt']
    start_minute_opt = config_dict['start_minute_opt']
    n_hr_test_horizon = config_dict['n_hr_test_horizon']
    ub_multiplier_list = config_dict['ub_multiplier_list']
    region_ = config_dict['region_']
    pad_str = config_dict['pad_str']
    n_hours = config_dict['n_hours']
    start_hour = config_dict['start_hour']
    interval_t = config_dict['interval_t']
    analysis_month = config_dict['analysis_month']
    analysis_day = config_dict['analysis_day']
    num_paths_temp = config_dict['num_paths_temp']
    incentive_list_available = config_dict['incentive_list_available']
    incentive_list_temp = config_dict['incentive_list_temp']
    new_file_temp = config_dict['new_file_temp']
    theta_temp = config_dict['theta_temp']
    folder_DPFE = config_dict['folder_DPFE']
    MIPGap_ = config_dict['MIPGap_']
    tot_budget_ = config_dict['tot_budget_']
    plot_ = config_dict['plot_']
    OD_add = config_dict['OD']
    one_arrival = config_dict['one_arrival']

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


    finish_hour_opt = start_hour_opt + n_hr_test_horizon
    # Check day
    analysis_date = datetime.date(random.randint(2018, 2018), random.randint(analysis_month, analysis_month), random.randint(analysis_day, analysis_day))  # !!
    i_str_available = '_' + '_'.join(str(x) for x in incentive_list_available) + '_'
    i_str_temp = '_' + '_'.join(str(x) for x in incentive_list_temp) + '_'
    finish_hour = start_hour + n_hours
    n_times_per_hr = int(60 / interval_t)
    h_start_test_temp = n_times_per_hr * (start_hour_opt - start_hour) + math.floor(start_minute_opt/interval_t)
    num_interval_temp = int(n_times_per_hr * n_hours)  # 7 hours, 12 5-minute intervals in each hour >> data 10AM-5PM
    analysis_start_time_temp = datetime.time(start_hour, 0, 0)  # !! Starting time = 3 AM # !!
    time_interval_temp = datetime.timedelta(minutes=interval_t)  # !! Time interval = 5 minutes

    theta_temp_str = "{:.0e}".format(theta_temp)

    OD_est_file_add_temp = folder_DPFE + '/Q_vector/' + new_file_temp + '/python3/2018-05-01.pickle'  # 10AM-5PM
    od_list_address = 'data/graph/' + region_ + '/my_od_list_' + region_ + '_original' + OD_add + '.pickle'
    graph_address = 'data/graph/' + region_ + '/my_graph_' + region_ + '_original.gpickle'
    pck_address_ub_temp = 'data/capacity/' + region_ + '/Mar2May_2018_new_5-22_link_capacity_' + region_ + '.pickle'  # !!
    pck_address_s_0_temp = 'data/capacity/' + region_ + '/Mar2May_2018_new_5-22_link_s_0_' + region_ + '.pickle'  # !!
    plot_address1 = 'plot'
    check_exists(add_=plot_address1, create_=True)
    # plot_address2 = 'plot/opt'
    # check_exists(add_=plot_address2, create_=True)
    # plot_address3 = 'plot/q_time'
    # check_exists(add_=plot_address3, create_=True)
    # plot_address4 = 'plot/VolumeVS.Capacity'
    # check_exists(add_=plot_address4, create_=True)
    # plot_address5 = 'plot/SpeedVS.S_0'
    # check_exists(add_=plot_address5, create_=True)
    # plot_address6 = 'plot/random'
    # check_exists(add_=plot_address6, create_=True)
    # plt_opt_address = 'plot/opt/' + region_ + i_str_available
    # check_exists(add_=plt_opt_address, create_=True)
    # q_plot_file = 'plot/q_time/' + region_ + i_str_available
    # check_exists(add_=q_plot_file, create_=True)

    gurobi_file_address0 = 'gurobi_files'
    check_exists(add_=gurobi_file_address0, create_=True)
    gurobi_file_address = 'gurobi_files/' + region_ + i_str_available + '_' + str(start_hour) + '-' + str(finish_hour) + pad_str + '_theta'+theta_temp_str
    check_exists(add_=gurobi_file_address, create_=True)
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

    # Loading the capacity of links
    pck_address = 'data/capacity/' + region_ + '/Mar2May_2018_new_5-22_link_capacity_' + region_ + '.pickle'
    with open(pck_address, 'rb') as f:
        link_cap_dict = pickle.load(f)

    tt_0_hour_add = "data/capacity/" + region_ + "/Mar2May_2018_new_5-22_link_tt_0_hours_" + region_ + ".pickle"
    with open(tt_0_hour_add, 'rb') as f:
        tt_0_BPR_hour_link = pickle.load(f)
    tt_0_BPR_hour_link_dict = dict()
    for index, row in tt_0_BPR_hour_link.iterrows():
        tt_0_BPR_hour_link_dict[int(row['link'])] = row['tt0_hours']

    # Load list of OD pairs
    with open(od_list_address, 'rb') as handle:  # !!
        (O_list, D_list) = pickle.load(handle, encoding='latin1')
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

                    length_path = 0
                    for iter_edge in range(len(path_temp) - 1):
                        length_path += \
                        G_temp.adj[path_temp[iter_edge]][path_temp[iter_edge + 1]]['length']
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
    print(f'Number of links: {num_link}')
    # print(f'Number of nodes: {}')
    print(f'Number of ODs: {num_OD}')
    print(f'Number of paths: {num_path}')

    assert (len(path_list) == num_path)

    # %%
    # ################################################ OD estimation ######################################
    start_time = time.time()
    OD_est_file_add = OD_est_file_add_temp

    with open(OD_est_file_add, 'rb') as f:
        q0 = pickle.load(f, encoding="latin1")
        # q_temp = np.array([math.floor(q_element) for q_element in q0])
        q_temp1 = np.array([round(q_element) for q_element in q0])

    q_temp2 = q_temp1.astype(int)

    # ? hours
    num_interval = num_interval_temp
    # Row >> time intervals, columns >> OD pairs
    q_temp3 = q_temp2.reshape(num_interval, -1)
    q = pd.DataFrame(q_temp3, copy=True)

    ##### Create ROW indecies of dataframe in Pandas
    month_dict = dict()
    # Business days of May 2018
    # month_dict[5] = [1, 2, 3, 4, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 29, 30, 31]  # !!
    month_dict[analysis_month] = [analysis_day]  # !!
    analysis_start_time = analysis_start_time_temp
    time_interval = time_interval_temp
    date_need_to_finish = list()
    for iter_month in month_dict.keys():
        for iter_day in month_dict[iter_month]:
            for h in range(num_interval):
                date_temp = datetime.date(2018, iter_month, iter_day)
                time_basis = (datetime.datetime.combine(date_temp, analysis_start_time) + h * time_interval).time()
                cur_date_time = datetime.datetime.combine(date_temp, time_basis)
                # single_date = cur_date_time.date()
                date_need_to_finish.append(cur_date_time)
    q.index = date_need_to_finish
    if plot_:
        fig0 = plt.figure()
        x_ = (np.array(list(range(0, n_hours * n_times_per_hr))) / float(n_times_per_hr)) + start_hour
        plt.plot(x_, np.sum(q.values, axis=1))
        plt.xlabel('Time (hour)')
        plt.ylabel('Total number of arriving drivers')
        plt.show()
        # fig0.savefig(q_plot_file + '/' + str(start_hour) + '_' + str(finish_hour) + '_AVG' + str(interval_t) + pad_str + '_theta' + theta_temp_str + '_new.png')


    h_start_test = h_start_test_temp
    test_horizon = n_hr_test_horizon*n_times_per_hr
    # q for only 12 horizons
    date_need_to_finish_test = list()
    for iter_month in month_dict.keys():
        for iter_day in month_dict[iter_month]:
            for h in range(test_horizon):
                date_temp = datetime.date(2018, iter_month, iter_day)  # !!
                time_basis = (datetime.datetime.combine(date_temp, analysis_start_time) + (h + h_start_test) * time_interval).time()
                cur_date_time = datetime.datetime.combine(date_temp, time_basis)
                # single_date = cur_date_time.date()
                date_need_to_finish_test.append(cur_date_time)
    q_test = q.loc[date_need_to_finish_test, :]

    n_drivers = [sum(temp_row) for temp_row in q_test.values]
    for iter in range(len(date_need_to_finish_test)):
        print(f"Total number of drivers at time {str(date_need_to_finish_test[iter])}: {n_drivers[iter]}")

    if plot_:
        x_2 = (np.array(list(range(0, math.floor(test_horizon/n_times_per_hr) * n_times_per_hr))) / float(n_times_per_hr)) + h_start_test_temp/n_times_per_hr + start_hour
        plt.plot(x_2, np.sum(q_test.values, axis=1))
        plt.xlabel('Time (hour)')
        plt.ylabel('Total number of arriving drivers')
        plt.title('')
        plt.show()

    time_temp_ = (time.time() - start_time) / 60
    print('Time of processing OD estimation vector (q): %f minutes' % time_temp_)
    # ################################################ Speed prediction ######################################
    print('Speed prediction, start')
    # # #### PREDICTION DATA ####
    # # D_in = 36
    # # H1 = 200
    # # H2 = 100
    # # H3 = 50
    # # D_out = 12
    # # N = 256
    # # data_name = 'simple_NN'
    # # test_every_n_epochs_ = 25
    # # # alpha_set = [2]
    # # # t_set = [5]
    # # # batch_percentage = 10
    # # MSE_ = True
    # # # Multiplier of the regularizer
    # # w_regul = 0
    # # regul = '_L2_regul_'
    # # gpu_ = False
    # # info = {'D_in': D_in, 'D_out': D_out, 'N': N, 'H1': H1, 'H2': H2, 'H3': H3, 'MSE_': MSE_, 'w_regul': w_regul,
    # #         'test_n_epochs': test_every_n_epochs_, 'data_name': data_name}
    # # # GPU or CPU
    # # if gpu_:
    # #     dtype = torch.cuda.FloatTensor
    # # else:
    # #     dtype = torch.FloatTensor
    # #
    # # if H1 == 0:
    # #     model = SimpleRegression(D_in, D_out)
    # # elif H2 == 0:
    # #     model = TwoLayerNet(D_in, H1, D_out)
    # # elif H3 == 0:
    # #     model = ThreeLayerNet(D_in, H1, H2, D_out)
    # # elif H3 > 0:
    # #     model = FourLayerNet(D_in, H1, H2, H3, D_out)
    # #
    # # # Load data and model
    # # data_file_train_ = 'mar2apr_2018_hist1_12_pred1_12_1PM_7PM_region_x1'
    # # data_file_test_ = 'may_2018_hist1_12_pred1_12_1PM_7PM_region_x1_sub1'
    # # test_x, test_y = data_loader.data_loader_f_test(data_file_test_)
    # # load_add = address.model_f(info, gpu_, 'saved_models', data_file_train_) + '_inference.pth'
    # # # The modules 'model_trainer_nn' and 'model_trainer_eval' are for simple_NN
    # # model.load_state_dict(torch.load(load_add))
    # # model.eval()
    # # # Prediction of speed
    # # speed_pred = model(test_x.type(dtype))
    # # # Tensor to numpy array
    # # speed_data = speed_pred.data.cpu().numpy()
    # # speed_data = pd.DataFrame(speed_data)
    # # # data_file_test_ = 'may_2018_hist1_12_pred1_12_1PM_7PM_region_x1_sub1'
    # # data_file_test_ = 'may_2018_hist1_12_pred1_12_5AM_8AM_region_x1_sub1_exception'
    # # _, speed_data = data_loader.data_loader_f_test(data_file_test_)
    # # speed_data = pd.DataFrame(speed_data)
    # #
    # # # inf_ data:ID, day, code
    # # # inf_file = 'inf_0_60_hist60_NN_may_2018_1PM_7PM_region_x1_sub1_test'
    # # inf_file = 'inf_0_60_hist60_NN_may_2018_5AM_8AM_region_x1_sub1_test_exception'
    # # inf_ = data_loader.data_loader_f(inf_file, False, False)
    #
    # # #### ORIGINAL DATA ####
    # data_file_test_ = 'may_2018_hist1_12_pred1_12_5AM_8AM_region_x1_sub1_OptModel'
    # speed_data = data_loader.data_loader_f(data_file_test_,  shuffle_=False, skip_header_=False)
    # speed_data = pd.DataFrame(speed_data)
    #
    # # inf_ data:ID, day, code
    # inf_file = 'inf_0_60_hist60_NN_may_2018_5AM_8AM_region_x1_sub1_test_OptModel'
    # inf_ = data_loader.data_loader_f(inf_file,  shuffle_=False, skip_header_=False)
    #
    # analysis_start_time2 = datetime.time(0, 0, 0)  # !! (int(inf_[iter_, 2])-1) will define the time
    # time_interval = datetime.timedelta(minutes=5)  # !! Time interval = 5 minutes # !!
    # date_need_to_finish = list()
    # for iter_ in range(inf_.shape[0]):
    #     # Test data is for May (5th month)
    #     date_temp = datetime.date(2018, 5, int(inf_[iter_, 1]))
    #     time_basis = (datetime.datetime.combine(date_temp, analysis_start_time2) + (int(inf_[iter_, 2])-1) * time_interval).time()
    #     cur_date_time = datetime.datetime.combine(date_temp, time_basis)
    #     date_need_to_finish.append(cur_date_time)
    # speed_data.index = date_need_to_finish
    print('Speed prediction, end')

    # ###################################### Plot the speed and volume data ########################################
    print('Plot the speed and volume data, start')
    # interval_t_cap = 5
    # n_times_per_hr_cap = int(60 / interval_t_cap)
    # start_hour_cap = 5
    # finish_hour_cap = 22
    # spd_add = 'data/speed_volume/Mar2May_2018_AVG' + str(interval_t_cap) + '_' + str(start_hour_cap) + '-' + str(finish_hour_cap) + '_' + region_ + '_with_linkID/my_link_avg_spd_data_AVG' + str(interval_t_cap) + 'min_' + str(start_hour_cap) + '-' + str(finish_hour_cap) + '_' + region_ + '.pickle'
    # with open(spd_add, 'rb') as f:
    #     spd_data_link = pickle.load(f)
    # volume_add = 'data/speed_volume/Mar2May_2018_AVG' + str(interval_t_cap) + '_' + str(start_hour_cap) + '-' + str(finish_hour_cap) + '_' + region_ + '_with_linkID/my_link_avg_count_data_AVG' + str(interval_t_cap) + 'min_' + str(start_hour_cap) + '-' + str(finish_hour_cap) + '_' + region_ + '.pickle'
    # with open(volume_add, 'rb') as f:
    #     volume_data_link = pickle.load(f)
    #
    # # Upper bound of capacity of links
    # pck_address = pck_address_ub_temp
    # ub_ = data_loader.data_loader_f(pck_address, shuffle_=False, skip_header_=False, name_=False)
    # l_ub_keys = list(ub_.keys())
    #
    # time_s = 0
    # time_f = (finish_hour_cap-start_hour_cap)*n_times_per_hr_cap
    # time_tot = time_f - time_s
    # t = [datetime.time(start_hour_cap + int(iter / n_times_per_hr_cap), iter % n_times_per_hr_cap * interval_t_cap, 00) for iter in range(time_tot)]  # !! 5 AM - 10 PM, Python 3
    # # x_axis_temp = []
    # # for iter_ in range(time_interval_t):
    # #     x_axis_[iter_]
    # #     x_axis_temp.append(x_axis_[iter_].hour*100 + x_axis_[iter_].minute)
    # str_date_temp = str(analysis_date)
    # x_ticks = volume_data_link[0].loc[analysis_date, t].index
    # for link_number in l_ub_keys:
    #     fig1 = plt.figure()
    #     plt.xticks(x_ticks)
    #     plt.xlabel('Time')
    #     plt.ylabel('Link volume')
    #     plt.title('Link %i, %s' % (link_number, str_date_temp))
    #     p1 = plt.plot(x_ticks, volume_data_link[link_number].loc[analysis_date, t])
    #     p2 = plt.plot(x_ticks, [ub_[link_number]]*len(x_ticks))
    #     p3 = plt.plot(x_ticks, [1.25*ub_[link_number]]*len(x_ticks))
    #     p4 = plt.plot(x_ticks, [1.5*ub_[link_number]]*len(x_ticks))
    #     p5 = plt.plot(x_ticks, [2*ub_[link_number]]*len(x_ticks))
    #     plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0]), ('Volume data', 'Volume capacity', '1.25*Volume capacity', '1.5*Volume capacity', '2*Volume capacity'))
    #     plt.locator_params(axis='x', nbins=8)
    #     plt.show()
    #     fig1.savefig(v_cap_plot_file + '/Link_' + str(link_number) + '_' + str_date_temp + '.png')
    #
    #
    #
    # # Free flow speed of links
    # pck_address = pck_address_s_0_temp
    # s_0_ = data_loader.data_loader_f(pck_address, shuffle_=False, skip_header_=False, name_=False)
    # l_s_0_keys = list(s_0_.keys())
    #
    # time_s = 0
    # time_f = (finish_hour_cap-start_hour_cap)*n_times_per_hr_cap
    # time_tot = time_f - time_s
    # t = [datetime.time(start_hour_cap + int(iter / n_times_per_hr_cap), iter % n_times_per_hr_cap * interval_t_cap, 00) for iter in range(time_tot)]  # !! 5 AM - 10 PM, Python 3
    # # x_axis_temp = []
    # # for iter_ in range(time_interval_t):
    # #     x_axis_[iter_]
    # #     x_axis_temp.append(x_axis_[iter_].hour*100 + x_axis_[iter_].minute)
    # str_date_temp = str(analysis_date)
    # x_ticks = spd_data_link[0].loc[analysis_date, t].index
    # for link_number in l_s_0_keys:
    #     fig1 = plt.figure()
    #     plt.xticks(x_ticks)
    #     plt.xlabel('Time')
    #     plt.ylabel('Link speed')
    #     plt.title('Link %i, %s' % (link_number, str_date_temp))
    #     p1 = plt.plot(x_ticks, spd_data_link[link_number].loc[analysis_date, t])
    #     p2 = plt.plot(x_ticks, [s_0_[link_number]]*len(x_ticks))
    #     plt.legend((p1[0], p2[0]), ('Speed data', 'Speed free flow'))
    #     plt.locator_params(axis='x', nbins=8)
    #     plt.show()
    #     fig1.savefig(speed_s_0_plot_file +  '/Link_s_0_' + str(link_number) + '_' + str_date_temp + '.png')
    print('Plot the speed and volume data, start')

    # ###################################### Constraint coefficients: probability and beta #########################
    start_time = time.time()
    print('Start loading A')
    # incentive_list = incentive_list_temp
    # analysis_date = datetime.date(random.randint(2018, 2018), random.randint(5, 5), random.randint(1, 1))  # !!
    # date_str = analysis_date.strftime("%Y-%m-%d")
    # new_file = new_file_temp
    # folder_DPFE = folder_DPFE_temp
    # observe_index_N = np.load(os.path.join(folder_DPFE, "observe_index_N", new_file, "observe_index_N.npy"))  # !!
    # # x_o = np.load(os.path.join(folder_DPFE, 'X_vector', new_file, date_str + ".npy"))
    # r = joblib.load(os.path.join(folder_DPFE, "R_matrix", new_file, date_str+".pickle")).tocsr()
    # A_dict = dict()
    # for incentive in incentive_list:
    #     P = joblib.load(os.path.join(folder_DPFE, "P_matrix", new_file, date_str+"_"+str(incentive)+".pickle")).tocsr()
    #     # P_ = P.todense()
    #     # perc_zeros = sum(P_.T == 0)/P_.shape[1]*100
    #     # plt.xlabel('Row number')
    #     # plt.ylabel('Number of zeros')
    #     # plt.title('Matrix P, Percentage of zeros')
    #     # plt.hist(perc_zeros, bins = 100)
    #     # plt.show()
    #     # The data of observed data in DAR matrix
    #     A_dict[incentive] = np.array(r.dot(P).todense())[observe_index_N == 1, :]
    #
    # test_horizon = test_horizon_temp
    # A_dict_test = dict()
    # # for incentive in incentive_list:
    # #     A_dict_test[incentive] = A_dict[incentive][(h_start_test)*num_link:(test_horizon+h_start_test)*num_link,
    # #                              (h_start_test)*num_link:(test_horizon+h_start_test)*num_OD]
    # for incentive in incentive_list:
    #     A_dict_test[incentive] = A_dict[incentive][(h_start_test)*num_link:(test_horizon+h_start_test)*num_link,
    #                              (h_start_test)*num_OD:(test_horizon+h_start_test)*num_OD]
    #
    # # # Check zeros in A
    # # for incentive in incentive_list:
    # #     plt.xlabel('Row number')
    # #     plt.ylabel('Number of zeros')
    # #     plt.title('Matrix A, number of zeros')
    # #     plt.plot(sum(A_dict_test[incentive].T == 0))
    # #     plt.show()
    # #     break
    # #
    # #     for iter_1 in range(test_horizon-1):
    # #         x_ = A_dict_test[incentive][iter_1*num_link:(iter_1+1)*num_link, (iter_1+1)*num_OD:]
    # #         for iter_2 in range(len(x_)):
    # #             if all(list(x_[iter_2] == 0)):
    # #                 print((iter_1, iter_2))
    # #                 break
    # #
    # # # Check zeros in A
    # # for incentive in incentive_list:
    # #     plt.xlabel('Row number')
    # #     plt.ylabel('Number of zeros')
    # #     plt.title('Matrix A, number of zeros')
    # #
    # #     max_idx = np.argmax(A_dict_test[incentive], axis=1)
    # #     b = A_dict_test[incentive][::-1]
    # #     i = b.shape[1] - np.argmax(b, axis=1) - 1
    # #     plt.plot(i)
    # #     plt.show()
    # #     break
    #
    #
    # time_temp_ = (time.time() - start_time) / 60
    # print('Time of processing Constraint coefficients: %f minutes' % time_temp_)
    print('Finish loading A')

    # ################################################## TEST ###########################################
    print('Start TEST')

    # p_ = P.todense()
    # for iter in range(num_path):
    #     # if p_[iter, 3]>0:
    #     #     print(iter, ':', p_[iter, 0:num_OD])
    #     # print(iter, ':', p_[iter, :]>0)
    #     l_g0 = [i for i, val in enumerate(np.array(p_[iter, 0:num_OD])[0]) if val > 0]
    #     print(iter, ':', l_g0)
    #
    #
    # l_ = list()
    # for iter_0 in range(12):
    #     for iter in range(num_path):
    #         # if p_[iter, 3]>0:
    #         #     print(iter, ':', p_[iter, 0:num_OD])
    #         # print(iter, ':', p_[iter, :]>0)
    #         l_g0 = [i for i, val in enumerate(np.array(p_[iter + num_path * iter_0, :])[0]) if val > 0]
    #         print('Time %i, link %i (min = %i, max = %i): ' %((iter_0+1), (iter+1), (iter_0)*num_OD, (iter_0+1)*num_OD), l_g0)
    #         # print(iter, ':', l_g0)
    #         l_.append(l_g0)
    #
    # l_ = list()
    # for iter_0 in range(12):
    #     # print('iter_0:', iter_0)
    #     for iter in range(num_link):
    #         # if p_[iter, 3]>0:
    #         #     print(iter, ':', p_[iter, 0:num_OD])
    #         # print(iter, ':', p_[iter, :]>0)
    #         l_g0 = [i for i, val in enumerate(np.array(A_dict[0][iter_0*num_link+iter, :])) if val > 0]
    #         print('Time %i, link %i (min = %i, max = %i): ' %((iter_0+1), (iter+1), (iter_0)*num_OD, (iter_0+1)*num_OD), l_g0)
    #         l_.append(l_g0[-1])
    # # np.array(A_dict[0][1, :])
    #
    # r_ = r.todense()
    #
    # l_ = list()
    # for iter_0 in range(12):
    #     # print('iter_0:', iter_0)
    #     for iter in range(num_link):
    #         # if p_[iter, 3]>0:
    #         #     print(iter, ':', p_[iter, 0:num_OD])
    #         # print(iter, ':', p_[iter, :]>0)
    #         r_temp = np.array(r_[iter])[0]
    #         l_g0 = [i for i, val in enumerate(np.array(r_[iter_0*num_link+iter, :])[0]) if val > 0]
    #         print('Time %i, link %i (max = %i): ' %((iter_0+1), (iter+1), (iter_0+1)*num_path), l_g0)
    #         l_.append(l_g0[-1])
    print('Finish loading A')
    # ################################################## optimization model ###########################################
    print('Start LP')
    q_test_OD = list(q_test.columns)
    q_test_time = q_test.index
    q_test_idx2 = list(range(len(q_test.index)))

    incentive_list = incentive_list_temp

    # Upper bound of capacity
    pck_address_ub = pck_address_ub_temp  # !!
    ub_ = data_loader.data_loader_f(pck_address_ub, shuffle_=False, skip_header_=False, name_=False)

    h_start_test = h_start_test_temp
    date_str = analysis_date.strftime("%Y-%m-%d")
    new_file = new_file_temp
    folder_DPFE = folder_DPFE
    r = joblib.load(os.path.join(folder_DPFE, "R_matrix", new_file, date_str+".pickle")).tocsr()
    r_ = r.todense()

    P_dict = dict()
    # P_dict[0] = joblib.load(
    #     os.path.join(folder_DPFE, "P_matrix", new_file, date_str + "_0.pickle")).tocsr()
    try:
        P_dict[0] = joblib.load(os.path.join(folder_DPFE, "P_matrix_opt", new_file, date_str + "_0.pickle")).tocsr()
    except:
        # P_dict[0] = joblib.load(os.path.join(folder_DPFE, "P_matrix_opt", new_file, date_str + "_0_pck.pickle")).tocsr()
        with open(os.path.join(folder_DPFE, "P_matrix_opt", new_file, date_str + "_0_pck.pickle"), 'rb') as f:
            P_dict[0] = pickle.load(f, encoding='latin1').tocsr()
    P_dict[0] = P_dict[0].todense()
    for i_iter in range(len(incentive_list)):
        i_iter2 = i_iter + 1
        try:
            P_dict[i_iter2] = joblib.load(os.path.join(folder_DPFE, "P_matrix_opt", new_file, date_str+"_"+str(incentive_list[i_iter])+".pickle")).tocsr()
        except:
            # P_dict[i_iter2] = joblib.load(os.path.join(folder_DPFE, "P_matrix_opt", new_file, date_str + "_" + str(
            #     incentive_list[i_iter]) + ".pickle")).tocsr()
            with open(os.path.join(folder_DPFE, "P_matrix_opt", new_file, date_str + "_" + str(
                incentive_list[i_iter]) + "_pck.pickle"), 'rb') as f:
                P_dict[i_iter2] = pickle.load(f, encoding='latin1').tocsr()
        P_dict[i_iter2] = P_dict[i_iter2].todense()

    idx_x = [0]
    idx_x_temp = 0
    for rs in range(len(OD_paths)):
        n_path_temp = int(np.sum(num_path_v[rs]))
        idx_x_temp += n_path_temp
        idx_x.append(idx_x_temp)

    with open(mile_address, 'rb') as f:
        df_length_mile = pickle.load(f)
    length_mile = list(df_length_mile['length_mile'].values)
    length_km = m2km(length_mile)

    incentive_list_opt = [0] + incentive_list

    # If one_arrival == True >>
    if one_arrival:
        for iter in range(test_horizon-1):
            q_test.loc[q_test_time[iter+1], :] = 0

    for ub_multiplier_temp in ub_multiplier_list:
        dota_teams_row = list()
        l_result_obj = list()
        l_result = list()
        spent_budget_l = list()
        tot_budget_feasible = list()
        CE_result = list()

        for tot_budget_iter in tot_budget_:
            print(f"\n\nUB multiplier: {ub_multiplier_temp}, Total budget: {tot_budget_iter}")
            # print(f"\n\nThe ub multiplier: {ub_multiplier_temp}")
            start_time = time.time()
            # ## Model
            m = gp.Model('Incentive')

            m.Params.MIPGap = MIPGap_

            # ## Variables
            incentive_list = incentive_list_temp
            tuple_var = tuple([(time_h1, OD_, driver_, path_, i)
                               for time_h1 in range(test_horizon)
                               for OD_ in q_test_OD
                               for driver_ in range(q_test.loc[q_test_time[time_h1], OD_])
                               for path_ in range(num_path_v[OD_])
                               for i in range(len(incentive_list)+1)  # +1 for 0 incentive
                               ])
            s = m.addVars(tuple_var, vtype=GRB.BINARY, name='S')

            # ######################################Correct Link Capacity Constraint
            m.addConstrs((gp.quicksum(s[time_h1, OD_, driver_, path_, i+1] *
                                      (gp.quicksum(r_[link_ + num_link*(h_start_test+time_h2), path_0 + idx_x[OD_] + num_path*(h_start_test+time_h1)] *
                                      P_dict[i+1][path_0 + idx_x[OD_] + num_path*(h_start_test+time_h1), path_ + idx_x[OD_]]
                                                   for path_0 in range(num_path_v[OD_]) if path_0 != path_) +
                                      r_[link_ + num_link*(h_start_test + time_h2), path_ + idx_x[OD_] + num_path*(h_start_test+time_h1)] *
                                      P_dict[i+1][path_ + idx_x[OD_] + num_path*(h_start_test+time_h1), path_ + idx_x[OD_]])
                                      for time_h1 in range(time_h2+1)
                                      for OD_ in q_test_OD
                                      for driver_ in range(q_test.loc[q_test_time[time_h1], OD_])
                                      for path_ in range(num_path_v[OD_])
                                      for i in range(len(incentive_list)))
                                      +
                                      gp.quicksum(s[time_h1, OD_, driver_, path_2, 0] *
                                      gp.quicksum(r_[link_ + num_link*(h_start_test+time_h2), path_1 + idx_x[OD_] + num_path*(h_start_test + time_h1)] *
                                      P_dict[0][path_1 + idx_x[OD_] + num_path*(h_start_test + time_h1), OD_]
                                      for path_1 in range(num_path_v[OD_]))
                                      for time_h1 in range(time_h2+1)
                                      for OD_ in q_test_OD
                                      for driver_ in range(q_test.loc[q_test_time[time_h1], OD_])
                                      for path_2 in range(num_path_v[OD_]))
                                      <= round(ub_[link_key_list[link_]]*ub_multiplier_temp)
                                      for time_h2 in range(test_horizon)
                                      for link_ in range(num_link)), name='Link_cap')
            # ######################################
            # ######################################
            # ######################################

            m.addConstr((gp.quicksum(s[time_h1, OD_, driver_, path_, i] * incentive_list_opt[i]
                                 for time_h1 in range(test_horizon)
                                 for OD_ in q_test_OD
                                 for driver_ in range(q_test.loc[q_test_time[time_h1], OD_])
                                 for path_ in range(num_path_v[OD_])
                                 for i in range(len(incentive_list)+1))
                                 <= tot_budget_iter), name='Tot_budget_cap')

            m.addConstrs((gp.quicksum(s[time_h1, OD_, driver_, path_, i]
                                    for path_ in range(num_path_v[OD_])
                                    for i in range(len(incentive_list)+1))
                                    == 1
                                    for time_h1 in range(test_horizon)
                                    for OD_ in q_test_OD
                                    for driver_ in range(q_test.loc[q_test_time[time_h1], OD_])), name='Decision_cap')

            m.addConstrs((gp.quicksum(s[time_h1+1, OD_, driver_, path_, 0]
                                    for path_ in range(num_path_v[OD_]))
                                    == 1
                                    for time_h1 in range(test_horizon-1)
                                    for OD_ in q_test_OD
                                    for driver_ in range(q_test.loc[q_test_time[time_h1+1], OD_])), name='Decision_ADMM_fix')


            m.setObjective(60*(gp.quicksum(gp.quicksum(s[time_h1, OD_, driver_, path_, i + 1] *
                                      (gp.quicksum(tt_0_BPR_hour_link_dict[link_key_list[link_]] *
                                                   r_[link_ + num_link * (h_start_test + time_h2), path_0 + idx_x[
                                          OD_] + num_path * (h_start_test + time_h1)]
                                                   *
                                                   P_dict[i + 1][path_0 + idx_x[OD_] + num_path * (
                                                           h_start_test + time_h1), path_ + idx_x[OD_]]
                                                   for path_0 in range(num_path_v[OD_]) if path_0 != path_) +
                                       tt_0_BPR_hour_link_dict[link_key_list[link_]] *
                                       r_[link_ + num_link * (h_start_test + time_h2), path_ + idx_x[OD_] + num_path * (
                                               h_start_test + time_h1)] *
                                       P_dict[i + 1][
                                           path_ + idx_x[OD_] + num_path * (h_start_test + time_h1), path_ + idx_x[
                                               OD_]])
                                      for time_h1 in range(time_h2 + 1)
                                      for OD_ in q_test_OD
                                      for driver_ in range(q_test.loc[q_test_time[time_h1], OD_])
                                      for path_ in range(num_path_v[OD_])
                                      for i in range(len(incentive_list)))
                          +
                          gp.quicksum(s[time_h1, OD_, driver_, path_2, 0] *
                                      gp.quicksum(tt_0_BPR_hour_link_dict[link_key_list[link_]] *
                                                  r_[link_ + num_link * (h_start_test + time_h2), path_1 + idx_x[
                                          OD_] + num_path * (h_start_test + time_h1)] *
                                                  P_dict[0][
                                                      path_1 + idx_x[OD_] + num_path * (h_start_test + time_h1), OD_]
                                                  for path_1 in range(num_path_v[OD_]))
                                      for time_h1 in range(time_h2 + 1)
                                      for OD_ in q_test_OD
                                      for driver_ in range(q_test.loc[q_test_time[time_h1], OD_])
                                      for path_2 in range(num_path_v[OD_]))
                          for time_h2 in range(test_horizon)
                          for link_ in range(num_link))), GRB.MINIMIZE)

            m.optimize()
            time_temp_ = (time.time() - start_time) / 60
            print('\n\Time of processing optimization: %f minute\n\n' % time_temp_)

            mem_dict = psutil.virtual_memory()._asdict()

            status = m.status

            if status == GRB.UNBOUNDED:  # code GRB.UNBOUNDED = 5
                print('The model cannot be solved because it is unbounded')
                # sys.exit(0)

            if status == GRB.INF_OR_UNBD:  # code GRB.INF_OR_UNBD = 4
                print('Optimization was stopped with status %d' % status)

            if status == GRB.INFEASIBLE:  # code GRB.INFEASIBLE = 3
                print('Optimization was stopped with status %d' % status)
                break
                # sys.exit(0)

            if status == GRB.OPTIMAL:  # code GRB.OPTIMAL = 2
                print('The optimal objective (tt in minutes) is %g' % m.objVal)
                if save_model_:
                    opt_file_path = gurobi_file_address + '/m_' + str(start_hour_opt) + '-' + str(finish_hour_opt) + '_'+str(ub_multiplier_temp)+'ub_theta_'+theta_temp_str + '_i' + i_str_temp + '_TotB' + str(tot_budget_iter)
                    m.write(opt_file_path + '_new.lp')
                    m.write(opt_file_path + '_new.sol')

                constrs = m.getConstrs()

                tot_CE = 0
                tot_tt = 0
                for iter_t in range(test_horizon):
                    for iter_link in range(num_link):
                        # print("\niter_t:", iter_t)
                        # print("iter_link:", iter_link)
                        # print("index:", iter_t*test_horizon + iter_link)

                        n_driver_temp = round(ub_[link_key_list[iter_link]] * ub_multiplier_temp) - (
                            constrs[iter_t * num_link + iter_link].slack)

                        if n_driver_temp < 0:
                            print('\n', iter_link)
                            print(iter_t)
                            print(iter_link * test_horizon + iter_t)
                            print(round(ub_[link_key_list[iter_link]] * ub_multiplier_temp))
                            print(round(constrs[iter_link * test_horizon + iter_t].slack))
                            print(n_driver_temp)

                        BPR_speed = bpr_f_speed(n_driver_temp, link_c_0_dict[link_key_list[iter_link]], ub_[link_key_list[iter_link]])
                        BPR_tt = length_km[link_key_list[iter_link]] * ((BPR_speed)**-1) * n_driver_temp
                        EF_temp = EF(BPR_speed) * length_km[link_key_list[iter_link]] * n_driver_temp
                        tot_CE += EF_temp
                        tot_tt += BPR_tt

                print(f"Total Carbon emission: {tot_CE}")

                print('The optimal objective (total travel time in hour)is %g' % (m.objVal / 60))
                print(f"Total travel time based on BPR: {tot_tt}")
                dota_teams_row.append("(" + str(ub_multiplier_temp) + ", " + str(tot_budget_iter) + ")")
                l_result_obj.append(m.objVal / 60)
                l_result.append(tot_tt)
                CE_result.append(tot_CE)


                temp_str = 'Tot_budget_cap'
                spent_budget_temp = tot_budget_iter - m.getConstrByName(temp_str).Slack
                print(f"Total budget: {tot_budget_iter}, Spent budget: {spent_budget_temp}")
                spent_budget_l.append(spent_budget_temp)
                tot_budget_feasible.append(tot_budget_iter)


                # Number of incentivized drivers:
                n_i_dict = dict()
                n_i_dict[0] = 0
                for i_temp in range(len(incentive_list)):
                    n_i_dict[i_temp + 1] = 0
                tot_n_driver = 0
                for v in m.getVars():
                    var_val = v.x
                    if var_val == 1:
                        i_temp = int(v.varName[-2])
                        # print(i_temp)
                        n_i_dict[i_temp] += 1
                        tot_n_driver += 1
                        # print(v)
                print(f"Total number of drivers: {tot_n_driver}")
                i_list_temp = [0] + incentive_list
                for i, (j, k) in enumerate(n_i_dict.items()):
                    print(f"{k} drivers received ${i_list_temp[j]} incentive")


        if l_result_obj:
            data_result = np.concatenate((np.expand_dims(np.array(l_result_obj), axis=1),
                                          np.expand_dims(np.array(l_result), axis=1),
                                          np.expand_dims(np.array(CE_result), axis=1),
                                          np.expand_dims(np.array(tot_budget_feasible), axis=1),
                                          np.expand_dims(np.array(spent_budget_l), axis=1)),
                                         axis=1)

            # for i1, j1 in enumerate(data_result):
            #     for i2, j2 in enumerate(j1):
            #         data_result[i1][i2] = round(data_result[i1][i2] * 1000000.) / 1000.

            print('\n\n')
            dota_teams_column = ["tt obj", "tt result", "CE", "Total budget", "Spent budget"]
            format_row = "{:>25}" * (len(dota_teams_column) + 1)
            print(format_row.format("", *dota_teams_column))
            for team, row in zip(dota_teams_row, data_result):
                print(format_row.format(team, *row))
        else:
            print('No result!')
        print('\n\n\n\n')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename',
                        default='YAML/region_z.yaml',
                        type=str,
                        help='Configuration filename for the region.')
    args = parser.parse_args()
    main(args)
