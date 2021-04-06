import numpy as np
import networkx as nx
import os
import pickle
import matplotlib
import matplotlib.pyplot as plt
import csv
import pandas as pd
import datetime
from data_loader import *
import data_loader
import googlemaps
from itertools import tee
import argparse
import yaml


def AM_PM_f(t):
    if t > 12:
        return str(t-12) + 'PM'
    elif t == 12:
        return '12PM'
    else:
        return str(t) + 'AM'


def hr_str(start_hr, finish_hr, AM_PM):
    if AM_PM:
        return AM_PM_f(start_hr), AM_PM_f(finish_hr)
    else:
        return str(start_hr), str(finish_hr)

def find_dist(adj_matrix_O_D, link_config):
    # Perform request to use the Google Maps API web service
    API_key = 'AIzaSyCGSGb99UKgvvK_MtadNufe5_WCb_4PyuI' # enter Google Maps API key
    gmaps = googlemaps.Client(key=API_key)
    # Empty list - will be used to store calculated distances
    list_ = []
    # Loop through each row in the adjacency matrix
    row_size_adj = adj_matrix_O_D.shape[0]
    for iter_ in range(row_size_adj):
        # Assign latitude and longitude as origin_point
        temp_idx = link_config['link_id'].astype(int).isin([adj_matrix_O_D.loc[iter_, 'origin'].astype(int)]) # !! Name of the column adj_matrix_O_D.loc[iter_, 'origin']]
        # temp_idx = link_config[:,0]==adj_matrix_O_D[iter_, 1]
        LatOrigin = list(link_config.loc[temp_idx, 'latitude'])
        LongOrigin = list(link_config.loc[temp_idx, 'longitude'])
        origin_ = (LatOrigin[0], LongOrigin[0])
        # print(origin_)
        # Assign latitude and longitude of the adjacent node as the destination_point
        temp_idx = link_config['link_id'].astype(int).isin([adj_matrix_O_D.loc[iter_, 'destination'].astype(int)]) # !! Name of the column adj_matrix_O_D.loc[iter_, 'destination']]
        # temp_idx = link_config[:,0]==adj_matrix_O_D[iter_, 2]
        LatDest = list(link_config.loc[temp_idx, 'latitude'])
        LongDest = list(link_config.loc[temp_idx, 'longitude'])
        destination_ = (LatDest[0], LongDest[0])
        # print(destination_, '\n\n')
        # Pass origin_ and destination_ variables to distance_matrix function
        # output in METERS
        result = gmaps.distance_matrix(origin_, destination_, mode='driving')["rows"][0]["elements"][0]["distance"]["value"]
        # print('\n', result)
        # Append result to list
        list_.append(result)
        # break
    return list_


def main(args):
    # ################################################# Initialization ##########################################
    # config_filename = 'YAML_graph/region_y2.yaml'
    with open(args.config_filename, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    region_input = config_dict['region_input']
    region_output = config_dict['region_output']
    sub_ = config_dict['sub_']
    region_of_subregion_ = config_dict['region_of_subregion_']
    interval_t = config_dict['interval_t']
    AM_PM = config_dict['AM_PM']
    start_hr_0 = config_dict['start_hr_0']
    finish_hr = config_dict['finish_hr']
    pad_ = config_dict['pad_']
    pad_hr = config_dict['pad_hr']
    data_year = config_dict['data_year']
    months_ = config_dict['months_']


    col_width = max(len(word) for word in config_dict.keys()) + 4  # padding
    for a, b in config_dict.items():
        print "".join([a.ljust(col_width), str(b)])

    # # ### Region
    # sub_ = False
    #
    # region_input = 'region_x4_modified'
    # # region_output = 'region_x4_modified'
    # region_output = 'region_x4_modified2'

    # region_ = 'region_x2_sub2'
    if not sub_:
        region_of_subregion_ = 'region_x2'
    #
    # # ### Time
    # # interval_t = 5
    # interval_t = 15

    n_times_per_hr = int(60/interval_t)
    #
    # AM_PM = False
    # start_hr_0 = 5  # 5 AM
    # finish_hr = 10  # 10 AM
    #
    # pad_ = True
    # pad_hr = 2
    if pad_:
        pad_str = '_pad'
        start_hr = start_hr_0 - pad_hr  # 5 AM
    else:
        pad_str = ''
        start_hr = start_hr_0

    start_hr_str, finish_hr_str = hr_str(start_hr_0, finish_hr, AM_PM)

    time_basis_temp = datetime.time(start_hr, 0, 0)  # !! 5 AM

    # data_year = 2018
    # months_ = 'Mar2May'

    graph_folder = 'data/graph/' + region_output
    if not os.path.exists(graph_folder):
        print('Folder does NOT exist: \n' + graph_folder)
    else:
        print('Folder exists: \n' + graph_folder)

    traffic_data_folder_input = 'data/speed_volume' + '/' + str(months_) + '_' + \
                          str(data_year) + '_' + region_input + '_AVG' + str(interval_t) + '_' + start_hr_str + '-' + \
                          finish_hr_str + '_with_linkID' + pad_str
    traffic_data_folder_output = 'data/speed_volume' + '/' + str(months_) + '_' + \
                          str(data_year) + '_' + region_output + '_AVG' + str(interval_t) + '_' + start_hr_str + '-' + \
                          finish_hr_str + '_with_linkID' + pad_str
    if not os.path.exists(traffic_data_folder_input):
        print('Folder does NOT exist: \n' + traffic_data_folder_input)
    else:
        print('Folder exists: \n' + traffic_data_folder_input)
    if not os.path.exists(traffic_data_folder_output):
        print('Folder does NOT exist: \n' + traffic_data_folder_output)
        os.mkdir(traffic_data_folder_output)
        print('Folder created:\n %s \n' % (traffic_data_folder_output))
    else:
        print('Folder exists: \n' + traffic_data_folder_output)
    # ################################################# Setting ################################################
    # Create ROW indecies of dataframe in Pandas
    month_dict = dict()
    # Business days of March 2018
    month_dict[3] = [1, 2, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 26, 27, 28, 29, 30]  # !!
    # Business days of April 2018
    month_dict[4] = [2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 23, 24, 25, 26, 27, 30]  # !!
    # Business days of May 2018
    month_dict[5] = [1, 2, 3, 4, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 29, 30, 31]  # !!

    # ################################################# Distance ################################################
    # Only find the distance of sensors in the adjacency matrix
    # import the sensors configuration data

    config_address = 'data/graph/2018_Feb2Nov_location_table.csv' # It has 3 columns: 1. link_id, 2. latitude, 3. longitude
    link_config = pd.read_csv(config_address, encoding="UTF-8", index_col=False)

    # import the sensors adjacency matrix !!
    # Adjacency matrix of region_x2_fixed.JPG @ "C:\Project\_assemble\shp_files\region_x2_original"
    # adj_address0 = 'data/graph/AdjMatrix_region_x2_original'
    adj_address0 = graph_folder + '/AdjMatrix_' + region_input + '_original'
    adj_address = adj_address0 + '.csv'
    adj_matrix = pd.read_csv(adj_address, index_col=False)
    adj_matrix = adj_matrix.reset_index(drop=True)
    # idx_dup = adj_matrix.duplicated()
    # adj_matrix = adj_matrix.loc[~idx_dup, :]
    # adj_matrix = adj_matrix.reset_index(drop=True)

    # adj_matrix.rename(columns={0:'origin', 1:'destination', 2:'direction'}, inplace=True)
    adj_matrix_O_D = adj_matrix[['origin', 'destination']]

    dist_l = find_dist(adj_matrix, link_config)
    # #### In case I considered both directions of O to D and D to O in my adjacency matrix
    adj_address0_dist = graph_folder + '/AdjMatrix_' + region_output + '_original'
    result_arr = pd.DataFrame(dist_l, columns=['distance'])
    adj_matrix2 = pd.concat([adj_matrix_O_D, result_arr], axis=1)
    adj_matrix2.index = range(adj_matrix2.shape[0])
    adj_address2 = adj_address0_dist + '_Distance.pickle'
    data_loader.data_pickle(adj_matrix2, adj_address2)
    adj_address2_csv = adj_address0_dist + '_Distance.csv'
    adj_matrix2.to_csv(adj_address2_csv)

    # ################################################ REGION X? ###################################################### Graph
    if not sub_:
        # Create the Graph (with new adjacency matrix)
        # adj_address_csv = 'data/graph/AdjMatrix_region_x2_original.csv'
        adj_address_csv = graph_folder + '/AdjMatrix_' + region_input + '_original.csv'
        adj_matrix = pd.read_csv(adj_address_csv, encoding='UTF-8', index_col=False)
        adj_matrix = adj_matrix.reset_index(drop=True)
        # idx_dup = adj_matrix.duplicated()
        # adj_matrix = adj_matrix.loc[~idx_dup, :]
        # adj_matrix = adj_matrix.reset_index(drop=True)


        # # Adding the column of ID
        dist_address = graph_folder + '/AdjMatrix_' + region_output + '_original_Distance.pickle'
        dist_sensors = data_loader.data_loader_f(dist_address, shuffle_=False, skip_header_=False)
        dist_sensors = dist_sensors.reset_index(drop=True)

        # # Remove duplicated rows of adj_region
        # idx_dup = dist_sensors.duplicated()
        # dist_sensors = dist_sensors.loc[~idx_dup, :]
        # dist_sensors = dist_sensors.reset_index(drop=True)

        length_ = list()
        # row_size_adj = adj_matrix.shape[0] # Previous Numpy version
        row_size_adj = adj_matrix.shape[0] # New Pandas version
        # Extract the distance of sensors of the adjacency matrix from all the vector of OD distances
        for iter_ in range(row_size_adj):
            temp_idx1 = dist_sensors['origin'].isin([adj_matrix.loc[iter_, 'origin'].astype(int)])
            temp_idx2 = dist_sensors.loc[temp_idx1, 'destination'].isin([adj_matrix.loc[iter_, 'destination'].astype(int)])
            dist_temp = np.array(dist_sensors.loc[temp_idx1, 'distance'][temp_idx2])[0]
            print dist_temp
            length_.append(dist_temp)
        length_ = pd.DataFrame(length_, columns=['length'])

        # Adding the column fft
        fft_ = pd.DataFrame(np.zeros(row_size_adj).reshape(row_size_adj, -1), columns=['fft']) # Pandas format

        # Find the sensors that are NODES
        link1 = adj_matrix['origin'] # New Pandas version
        link2 = adj_matrix['destination'] # New Pandas version
        unique_sensors = np.concatenate((link1, link2))
        unique_sensors = np.unique(unique_sensors)
        unique_sensors = unique_sensors.astype(int)
        # unique_sensors = NODES
        unique_sensors = unique_sensors[np.argsort(unique_sensors)]
        address_pickle = graph_folder + '/unique_sensors_' + region_output + '_original.pickle'
        pickle.dump(unique_sensors, open(address_pickle, "wb"))

        # Find the sensors that are on edges (links)
        adj_matrix_drop_dir = adj_matrix.drop('direction', axis=1)
        temp_links = list()
        for iter_ in range(adj_matrix_drop_dir.shape[0]):
            temp_ = np.array(adj_matrix_drop_dir.loc[iter_])
            temp_list = list(temp_[~np.isnan(temp_)].astype(int))
            temp_links = temp_links + temp_list
        # Get unique sensors in the links
        # unique_links = SENSORS ON EDGE
        unique_links = pd.DataFrame(list(set(temp_links)))

        DG = nx.DiGraph()
        DG.add_nodes_from(list(unique_sensors))

        for link in range(adj_matrix.shape[0]):
            DG.add_edge(adj_matrix.loc[link, 'origin'].astype(int), adj_matrix.loc[link, 'destination'].astype(int), ID=link,
                        length=length_.loc[link, 'length'], fft=fft_.loc[link, 'fft'])

        # graph_address = 'data/graph/my_graph_region_x2_original.gpickle'
        graph_address = graph_folder + '/my_graph_' + region_output + '_original.gpickle'
        nx.write_gpickle(DG, graph_address)

        # Saving the ID of link sensors of the region (all the sensors)
        # ID_address = 'data/graph/my_ID_region_x2_original.pickle'
        ID_address = graph_folder + '/my_ID_' + region_output + '_original.pickle'
        data_loader.data_pickle(unique_links, ID_address)
        ID_address_CSV = graph_folder + '/my_ID_' + region_output + '_original.csv'
        unique_links.to_csv(ID_address_CSV, header=False, index=False)

        od_list_common = pd.DataFrame(unique_sensors)
        od_list = [list(od_list_common[0].values), list(od_list_common[0].values)]
        # od_list_address = 'data/graph/my_od_list_region_x2_original.pickle'
        od_list_address = graph_folder + '/my_od_list_' + region_output + '_original.pickle'
        data_loader.data_pickle(od_list, od_list_address)


    else:
        # ###########################################     SUB?    ####################################################
        # Generate the data of link sensors and OD sensors for the selected subregion
        # Loading the ID of OD sensors of the selected region that are included in the dataset
        # region_x2_sub1_original includes 17 sensors
        selected_OD_list_address = graph_folder + '/selected_OD_' + region_output + '_original.csv'
        # selected_OD_list = pd.read_csv(selected_OD_list_address, encoding='UTF-8', index_col=False, header=None)
        selected_OD_list = pd.read_csv(selected_OD_list_address, index_col=False, header=None)
        # Finding the available IDs of selected region in the larger adjacency matrix
        address_pickle = graph_folder + '/unique_sensors_' + region_of_subregion_ + '_original.pickle'
        unique_sensors = data_loader.data_loader_f(address_pickle, shuffle_=False, skip_header_=False, name_=False)

        od_list_common = pd.merge(pd.DataFrame(unique_sensors), selected_OD_list, how='inner')
        od_list = [list(od_list_common[0].values), list(od_list_common[0].values)]
        od_list_address = graph_folder + '/my_od_list_' + region_output + '_original.pickle'
        data_loader.data_pickle(od_list, od_list_address)

        adj_matrix_selected = adj_matrix.copy(deep=True)
        adj_matrix_selected.drop(adj_matrix_selected.index, inplace=True)
        # Create the adjacency matrix of the subregion
        for iter_ in range(od_list_common.shape[0]):
            od_list_selected_ = list(od_list_common.loc[:, 0].values.astype(int))
            # Index of rows in the adjacency matrix that sensor od_list_common.loc[iter_, 0] is origin
            temp_idx_o_1 = adj_matrix['origin'].isin([od_list_common.loc[iter_, 0].astype(int)])
            temp_idx_o_2 = adj_matrix.loc[temp_idx_o_1, 'destination'].isin(od_list_selected_)
            adj_matrix_selected = adj_matrix_selected.append(adj_matrix.loc[temp_idx_o_1, :].loc[temp_idx_o_2, :], ignore_index=True)

            # WARNING: We don't need to find the selected IDs in the destination since it causes duplication
            # # Index of rows in the adjacency matrix that sensor od_list_common.loc[iter_, 0] is destination
            # temp_idx_d_1 = adj_matrix['destination'].isin([od_list_common.loc[iter_, 0].astype(int)])
            # temp_idx_d_2 = adj_matrix.loc[temp_idx_o_1, 'origin'].isin(od_list_selected_)
            # adj_matrix_selected = adj_matrix_selected.append(adj_matrix.loc[temp_idx_o_1, :].loc[temp_idx_o_2, :], ignore_index=True

        adj_address_selected_pickle = graph_folder + '/AdjMatrix_' + region_input + '_original.pickle'
        pickle.dump(adj_matrix_selected, open(adj_address_selected_pickle, "wb"))

        # Saving the ID of link sensors of the selected region that are included in the dataset
        # Find the sensors that are on edges (links)
        adj_matrix_selected_drop_dir = adj_matrix_selected.drop('direction', axis=1)
        temp_links = list()
        for iter_ in range(adj_matrix_selected_drop_dir.shape[0]):
            temp_ = np.array(adj_matrix_selected_drop_dir.loc[iter_])
            temp_list = list(temp_[~np.isnan(temp_)].astype(int))
            temp_links = temp_links + temp_list
        # Get unique sensors in the links
        # unique_links = SENSORS ON EDGE
        unique_links_selected = pd.DataFrame(list(set(temp_links)))

        # Find the sensors that are NODES
        link1 = adj_matrix_selected['origin'] # New Pandas version
        link2 = adj_matrix_selected['destination'] # New Pandas version
        unique_sensors_selected = np.concatenate((link1, link2))
        unique_sensors_selected = np.unique(unique_sensors_selected)
        unique_sensors_selected = unique_sensors_selected.astype(int)
        # unique_sensors_selected = NODES
        unique_sensors_selected = unique_sensors_selected[np.argsort(unique_sensors_selected)]


        dist_address = graph_folder + '/AdjMatrix_' + region_of_subregion_ + '_original_Distance.pickle'
        dist_sensors = data_loader.data_loader_f(dist_address, shuffle_=False, skip_header_=False)
        dist_sensors = dist_sensors.reset_index(drop=True)

        # # Remove duplicated rows of adj_region
        # idx_dup = dist_sensors.duplicated()
        # dist_sensors = dist_sensors.loc[~idx_dup, :]
        # dist_sensors = dist_sensors.reset_index(drop=True)

        row_size_adj_selected = adj_matrix_selected.shape[0] # New Pandas version

        length_ = list()
        # Extract the distance of sensors of the adjacency matrix from all the vector of OD distances
        for iter_ in range(adj_matrix_selected.shape[0]):
            temp_idx1 = dist_sensors['origin'].isin([adj_matrix_selected.loc[iter_, 'origin'].astype(int)])
            temp_idx2 = dist_sensors.loc[temp_idx1, 'destination'].isin([adj_matrix_selected.loc[iter_, 'destination'].astype(int)])
            dist_temp = np.array(dist_sensors.loc[temp_idx1, 'distance'][temp_idx2])[0]
            length_.append(dist_temp)
        length_ = pd.DataFrame(length_, columns=['length'])

        # Adding the column fft
        fft_ = pd.DataFrame(np.zeros(row_size_adj_selected).reshape(row_size_adj_selected, -1), columns=['fft']) # Pandas format

        # Save the graph
        DG = nx.DiGraph()
        DG.add_nodes_from(list(unique_sensors_selected))

        for link in range(adj_matrix_selected.shape[0]):
            DG.add_edge(adj_matrix_selected.loc[link, 'origin'].astype(int), adj_matrix_selected.loc[link, 'destination'].astype(int), ID=link,
                        length=length_.loc[link, 'length'], fft=fft_.loc[link, 'fft'])

        graph_address = graph_folder + '/my_graph_' + region_output + '_original.gpickle'
        nx.write_gpickle(DG, graph_address, protocol=2)

        # Saving the ID of link sensors of the region
        ID_address = graph_folder + '/my_ID_' + region_output + '_original.pickle'
        data_loader.data_pickle(unique_links_selected, ID_address)
        ID_address_CSV = graph_folder + '/my_ID_' + region_output + '_original.csv'
        unique_links_selected.to_csv(ID_address_CSV, header=False, index=False)

    # ###################################     Data      #######################################
    # Create the speed and volume data
    def data_days_f(time_basis_temp, data_year, month_dict):
        date_need_to_finish = list()
        for iter_month in month_dict.keys():
            for iter_day in month_dict[iter_month]:
                # print '\n'
                date_temp = datetime.date(data_year, iter_month, iter_day)  # !!
                time_basis = time_basis_temp
                cur_date_time = datetime.datetime.combine(date_temp, time_basis)
                # print cur_date_time
                single_date = cur_date_time.date()
                # print single_date
                date_need_to_finish.append(single_date)
        return date_need_to_finish

    date_need_to_finish = data_days_f(time_basis_temp, data_year, month_dict)


    # Load data
    # Sorted IDs
    # IDs = pd.read_csv('sorted_IDs_Mar2May_2018.csv', encoding='UTF-8', header=None, index_col=False) # !!
    # ID_address = 'my_ID.pickle' # sorted_IDs_Mar2May_2018_new
    ID_address = graph_folder + '/my_ID_' + region_output + '_original.pickle'
    IDs = data_loader.data_loader_f(ID_address, shuffle_=False, skip_header_=False) # !! New pickled version

    # Speed data
    speed_data = pd.read_csv(traffic_data_folder_input + '/speed_' + str(months_) + '_' + str(data_year) + '_new_' + start_hr_str + '-' + finish_hr_str + '_' + region_input + pad_str + '.csv',
                             encoding='UTF-8', header=None, index_col=False) # !!

    # Volume data
    volume_data = pd.read_csv(traffic_data_folder_input + '/volume_' + str(months_) + '_' + str(data_year) + '_new_' + start_hr_str + '-' + finish_hr_str + '_' + region_input + pad_str + '.csv',
                              encoding='UTF-8', header=None, index_col=False) # !!

    # Create COLUMN labels of dataframe in Pandas
    # num_time = 17*12  # 17 hours, intervals = 5 minutes >> 17*12 codes # !!
    num_time = int((finish_hr - start_hr)*n_times_per_hr)  # 17 hours, intervals = 5 minutes >> 17*12 codes # !!
    t = [datetime.time(start_hr + int(iter / n_times_per_hr), iter % n_times_per_hr * interval_t, 00) for iter in range(num_time)]  # Python 3


    def dict_data_IDs_f(data_, IDs_, date_need_to_finish_):
        data_dict_ = dict()
        for iter_ID in range(IDs_.shape[0]):
            ID_ = IDs_.loc[iter_ID, 0]
            # Speed data
            temp_idx_ = data_[0].isin([ID_.astype(int)])
            temp_data = data_.loc[temp_idx_, 1:]
            temp_data.index = date_need_to_finish_
            temp_data.columns = t
            data_dict_[ID_] = temp_data
        return data_dict_

    speed_dict = dict_data_IDs_f(speed_data, IDs, date_need_to_finish)
    volume_dict = dict_data_IDs_f(volume_data, IDs, date_need_to_finish)



    # Read adjacency matrix
    adj_address_csv = graph_folder + '/AdjMatrix_' + region_input + '_original.csv'
    adj_matrix_selected = pd.read_csv(adj_address_csv, encoding='UTF-8', index_col=False)
    # idx_dup = adj_matrix_selected.duplicated()
    # adj_matrix_selected = adj_matrix_selected.loc[~idx_dup, :]
    # adj_matrix_selected = adj_matrix_selected.reset_index(drop=True)
    # adj_address = 'data/graph/AdjMatrix_region_x2_sub1_original.pickle'
    # adj_matrix_selected = data_loader.data_loader_f(adj_address, shuffle_=False, skip_header_=False) # !! New pickled version
    adj_matrix_selected_drop_dir = adj_matrix_selected.drop('direction', axis=1)

    # ########   WHICH ONE??????????????????????????????????????????????

    # We should take the average of data of sensors on each link
    speed_dict_avg = dict()
    volume_dict_avg = dict()
    for iter_ in range(adj_matrix_selected_drop_dir.shape[0]):
        temp_ = np.array(adj_matrix_selected_drop_dir.loc[iter_])
        # temp_list = list(temp_[~np.isnan(temp_)].astype(int))
        temp_list = list(np.unique(temp_[~np.isnan(temp_)].astype(int)))
        temp_speed_data = pd.concat([speed_dict[temp_list[0]], speed_dict[temp_list[1]]])
        temp_volume_data = pd.concat([volume_dict[temp_list[0]], volume_dict[temp_list[1]]])
        for iter_link in range(len(temp_list)-2):
            temp_speed_data = pd.concat([temp_speed_data, speed_dict[temp_list[iter_link+2]]])
            temp_volume_data = pd.concat([temp_volume_data, volume_dict[temp_list[iter_link+2]]])
        temp_avg_speed_data = temp_speed_data.groupby(level=0).mean()
        temp_avg_volume_data = temp_volume_data.groupby(level=0).mean().apply(np.ceil)
        speed_dict_avg[iter_] = temp_avg_speed_data
        volume_dict_avg[iter_] = temp_avg_volume_data

    # ########   WHICH ONE??????????????????????????????????????????????

    # # We should take the average of data of sensors on each link
    # speed_dict_avg = dict()
    # volume_dict_avg = dict()
    # for iter_ in range(adj_matrix_selected_drop_dir.shape[0]):
    #     temp_ = np.array(adj_matrix_selected_drop_dir.loc[iter_])
    #     # temp_list = list(temp_[~np.isnan(temp_)].astype(int))
    #     temp_list = list(temp_[~np.isnan(temp_)].astype(int))
    #     temp_speed_data = pd.concat([speed_dict[temp_list[2]], speed_dict[temp_list[3]]])
    #     temp_volume_data = pd.concat([volume_dict[temp_list[2]], volume_dict[temp_list[3]]])
    #     for iter_link in range(len(temp_list)-4):
    #         temp_speed_data = pd.concat([temp_speed_data, speed_dict[temp_list[iter_link+4]]])
    #         temp_volume_data = pd.concat([temp_volume_data, volume_dict[temp_list[iter_link+4]]])
    #     temp_avg_speed_data = temp_speed_data.groupby(level=0).mean()
    #     temp_avg_volume_data = temp_volume_data.groupby(level=0).mean().apply(np.ceil)
    #     speed_dict_avg[iter_] = temp_avg_speed_data
    #     volume_dict_avg[iter_] = temp_avg_volume_data

    # Save the speed and volume data of edges in the graph
    # print 'Dumping average speed data'
    # data_pickle(speed_dict_avg, 'data/speed_volume/Mar2May_2018_AVG5_5AM-12PM_with_linkID/my_link_avg_spd_data_AVG5min_5AM-12PM_region_x2_sub1.pickle')
    # data_pickle(speed_dict_avg, 'data/speed_volume/Mar2May_2018_AVG5_10AM-17PM_with_linkID/my_link_avg_spd_data_AVG5min_10AM-17PM_region_x2_sub1.pickle')
    data_pickle(speed_dict_avg, traffic_data_folder_output + '/my_link_avg_spd_data_AVG' + str(interval_t) + 'min_' + start_hr_str + '-' + finish_hr_str + '_' + region_output + pad_str + '.pickle')

    # print 'Dumping average count data'
    # data_pickle(volume_dict_avg, 'data/speed_volume/Mar2May_2018_AVG5_5AM-12PM_with_linkID/my_link_avg_count_data_AVG5min_5AM-12PM_region_x2_sub1.pickle')
    # data_pickle(volume_dict_avg, 'data/speed_volume/Mar2May_2018_AVG5_10AM-17PM_with_linkID/my_link_avg_count_data_AVG5min_10AM-17PM_region_x2_sub1.pickle')
    data_pickle(volume_dict_avg, traffic_data_folder_output + '/my_link_avg_count_data_AVG' + str(interval_t) + 'min_' + start_hr_str + '-' + finish_hr_str + '_' + region_output + pad_str + '.pickle')



if __name__ == "__main__":
    # main()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename',
                        default='YAML_graph/region_z.yaml',
                        type=str,
                        help='Configuration filename for the region.')
    args = parser.parse_args()
    main(args)
