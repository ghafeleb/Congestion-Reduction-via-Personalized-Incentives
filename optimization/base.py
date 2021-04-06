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
from itertools import islice


class Link:
    def __init__(self, ID, length, fft):
        self.ID = ID
        self.length = length
        self.fft = fft


class Path:
    def __init__(self):
        self.node_list = None
        self.link_list = None
        self.cost = None
        self.p = None
        # self.length_ = None
        return

    # Construct the link_list attribute of the Path object from node_list attribute
    def node_to_list(self, G, link_dict):
        # self.length_ = list()
        if self.node_list == None:
            print("Nothing to convert")
            return
        # tmp is going to finally become link_list attribute
        tmp = list()
        # Loop in the number of links in the path (from O to D) - 1 >> -1 since we need i and i+1 so i+1 should not give error
        for i in range(len(self.node_list) - 1):
            try:
                # print("OK")
                # self.length_.append(G[self.node_list[i]][self.node_list[i+1]]["length"])
                # print("node_list[", i,"] and node_list[", i+1, "]:", self.node_list[i], self.node_list[i+1])
                link_ID = G[self.node_list[i]][self.node_list[i + 1]]["ID"]
                # if i==0:
                #     # if G[self.node_list[i]][self.node_list[i+1]]["length"]!=0:
                #     print("Weight of edge from O:", G[self.node_list[i]][self.node_list[i+1]]["length"])
                # elif i==len(self.node_list) - 1-1:
                #     # if G[self.node_list[i]][self.node_list[i+1]]["length"]!=0:
                #     print("Weight of edge to D:", G[self.node_list[i]][self.node_list[i + 1]]["length"]),

                # If the edge is not traversed before, we add its information to link_dict
                if link_ID not in link_dict.keys():
                    # print('if1')

                    # Construct a Link object
                    # print("Node ", self.node_list[i], " to ", self.node_list[i + 1])
                    # print("Link ID: ", link_ID, ", Length: ", G[self.node_list[i]][self.node_list[i + 1]][
                    #     "length"], ", fft:", G[self.node_list[i]][self.node_list[i + 1]]["fft"])
                    tmp_link = Link(link_ID, G[self.node_list[i]][self.node_list[i + 1]]["length"],
                                    G[self.node_list[i]][self.node_list[i + 1]]["fft"])
                    # print('if2')
                    tmp.append(tmp_link)
                    # print('if3')
                    link_dict[link_ID] = tmp_link
                else:
                    # print("\nThis link has been checked before, link_ID:", link_ID)
                    tmp.append(link_dict[link_ID])
                    # print('else')

            except:
                print("ERROR")
                print(self.node_list[i], self.node_list[i + 1])
        self.link_list = tmp


def k_shortest_paths(G, source, target, k, weight='length'):  # !! My code
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))  # !! My code


def get_finish_time(spd, length_togo, start_time, tmp_date):
    # basis = datetime.datetime.combine(tmp_date, datetime.time(0, 0, 0)) # !!
    # basis = datetime.datetime.combine(tmp_date, datetime.time(5, 0, 0)) # !!
    # basis = datetime.datetime.combine(tmp_date, datetime.time(15, 0, 0)) # !!
    basis = datetime.datetime.combine(tmp_date, datetime.time(14, 0, 0))  # !!
    # a function with lambda is defined, then all the values in spd.index inserted in this function using map(function, values)
    # Next line changes all the speed values in data to time_seq
    # spd: The row of the speed data at a specific day at a specific sensor
    # spd.index is the list of time intervals included in the column indecies of speed data in seconds >> 5AM = 18000
    time_seq = map(lambda x: (datetime.datetime.combine(tmp_date, x) - basis).total_seconds(), spd.index)
    # Create a float array of speed data
    data = np.array(spd.tolist()).astype(np.float)
    #     print data
    #     print time_seq
    # np.interp(x_tilt, x_data, f(x_data)): finds the linear interpolation between two consecutive points of (x_data, f(x_data)) if f() is a function,
    # in other words, it wants to find the value of f(x_tilt) with interpolation
    # Dividing by 1600 to convert meter to miles (?!)
    # Multiply by 3600 to convert time to seconds (?!)
    # Therefore: (meter/hour)*(miles/1600 meter)*(hour/3600 seconds) --> (miles/seconds)
    # And: (miles/hour)*(1600 meter/1 mile)*(1 hour/3600 seconds) --> (meters/seconds)
    # cur_spd = np.interp((datetime.datetime.combine(tmp_date, start_time) - basis).total_seconds(), time_seq, data) / 1600.0 * 3600.0 # !! His code
    cur_spd = np.interp((datetime.datetime.combine(tmp_date, start_time) - basis).total_seconds(), time_seq,
                        data) * 1600.0 / 3600.0  # !! My code
    try:
        # length_togo is the length of the link <<>> link_dict[ID].length
        # Using timedelta function to manipulate the time
        # We use ".time()" to convert seconds to time format > example: (basis + datetime.timedelta(seconds = 3600)).time() --> datetime.time(1, 0)
        # start_time + time needed to reach link
        # print "current speed: ", cur_spd
        # print "length_togo: ", length_togo
        new_start_time = (datetime.datetime.combine(tmp_date, start_time) + datetime.timedelta(
            seconds=length_togo / cur_spd)).time()
    #     print "need:", length_togo/cur_spd
    except:
        new_start_time = (datetime.datetime.combine(tmp_date, start_time) + datetime.timedelta(seconds=10)).time()  # !!
    return new_start_time


def get_arrival_time(start_time, link, spd_data, tmp_date, link_dict):
    link_to_pass = link

    # If link has 0 length >>> time of traversing is zero >>> return the time without any change
    if link_to_pass.length == np.float(0):
        return start_time

    # If the speed data of link is not available
    if link_to_pass.ID not in spd_data.keys():
        new_start_time = (datetime.datetime.combine(tmp_date, start_time) + datetime.timedelta(
            seconds=link_to_pass.fft)).time()
        return new_start_time

    # If the speed data of link is available
    try:
        # .loc[tmp_date] is the row of the speed at this specific day
        # spd_data[link_to_pass.ID] is the data of sensor with ID of checked link which is link_to_pass.ID
        spd = spd_data[link_to_pass.ID].loc[tmp_date]  # !!
    # If the speed data of link is not available
    except:
        new_start_time = (datetime.datetime.combine(tmp_date, start_time) + datetime.timedelta(
            seconds=link_to_pass.fft)).time()
        return new_start_time
    length_togo = link_to_pass.length

    # get_finish_time: gets the speed data of a link at a day, length and start time to find the finish_time. finish_time = time that we reach the destination
    # print "link.ID: ", link.ID
    new_start_time = get_finish_time(spd, length_togo, start_time, tmp_date)
    return new_start_time


def softmax(x, incentive, theta=1):
    #     print x
    """Compute softmax values for each sets of scores in x."""
    # Multiplication by theta to make the values of np.exp(.) more reasonable
    y = (np.copy(x)/60 * -0.086 + incentive * 0.7) * theta
    # print y
    p = np.minimum(np.maximum(np.exp(y), 1e-20), 1e20) / np.sum(np.minimum(np.maximum(np.exp(y), 1e-20), 1e20), axis=0)
    #     print p
    # If any element of p is Nan, return equal probablity for all the paths
    if np.isnan(p).any():
        p = np.ones(len(x)) / len(x)
    return p


def get_full_arrival_time(start_time, link_list, spd_data, tmp_date, link_dict, spd=None):
    #     if len(link_list) == 0:
    #         return start_time
    #     link_to_pass = link_list[0]
    #     if link_to_pass.length == np.float(0):
    #         link_list.pop(0)
    #         return get_full_arrival_time(start_time, link_list, spd_data, tmp_date, link_dict)
    #     if link_to_pass.ID not in spd_data.keys():
    #         link_list.pop(0)
    #         new_start_time = (datetime.datetime.combine(tmp_date, start_time) + datetime.timedelta(seconds = np.round(link_to_pass.fft))).time()
    #         return get_full_arrival_time(new_start_time, link_list, spd_data, tmp_date, link_dict)
    #     if type(spd) == type(None):
    #         spd = spd_data[link_to_pass.ID].loc[tmp_date]
    #     length_togo = link_to_pass.length
    #     new_start_time = get_finish_time(spd, length_togo, start_time, tmp_date)
    #     link_list.pop(0)
    arrival_time = copy.copy(start_time)
    for link in link_list:
        # Given the arrival_time to the last link, when is the departure time of the current link in the path?
        # Note that the we give the last arrival_time and we get the new one.
        arrival_time = get_arrival_time(arrival_time, link, spd_data, tmp_date, link_dict)
    return arrival_time


# tmp_date = datetime.date(2014, 1, 1)
# Generating route choice matrix
def get_P(N, spd_data, analysis_start_time, time_interval, tmp_date, path_list, OD_paths, incentive):
    # vector of number of paths between each OD pair
    num_path_v = [len(x) for x in OD_paths.itervalues()]
    # Total number of paths
    num_path = np.sum(num_path_v)
    OD_list = list(OD_paths.keys())
    num_OD = len(OD_list)
    row_list = list()
    col_list = list()
    data_list = list()
    for h in range(N):
        #         print h, N
        start_time = (datetime.datetime.combine(tmp_date, analysis_start_time) + h * time_interval).time()
        # First finding the probabilities and assigning them ti "p" attribute of the Path objects
        for (O, D), paths in OD_paths.iteritems():
            #         print (O,D)
            # List of cost of different paths between a specific OD pair
            cost_list = list()
            for path in paths:
                # Given start driving from origin at start_time, when do we reach the destination? ans: arrival_time
                arrival_time = get_full_arrival_time(start_time, path.link_list, spd_data, tmp_date, None)
                cost = (datetime.datetime.combine(tmp_date, arrival_time) - datetime.datetime.combine(tmp_date,
                                                                                                      start_time)).total_seconds()
                # "cost" is an attribute of Path objects
                path.cost = cost
                cost_list.append(cost)
            p_list = softmax(cost_list, incentive)
            # print "cost_list:", cost_list
            # print "p_list:", p_list
            for idx, path in enumerate(paths):
                # "p" is an attribute of Path objects
                path.p = p_list[idx]
        #         print p_list
        # Second, assigning the generate probabilities to the route choice matrix
        # OD pair rs
        for rs, (O, D) in enumerate(OD_list):
            # Path k of OD
            for k, path in enumerate(path_list):
                # Iterating over all the paths between (O, D)
                if k < np.sum(num_path_v[0:rs + 1]) and k >= np.sum(num_path_v[0:rs]):
                    # (h-1)*|K| + rs >> Paper, page 7, Table 2 >> OD pair rs at time interval h
                    x_loc = h * num_OD + rs
                    # (h-1)*|PI| + k >> Paper, page 7, Table 2 >> Path index at time interval h
                    y_loc = h * num_path + k
                    # Vector of probabilities of paths between OD
                    data = path.p
                    row_list.append(y_loc)
                    col_list.append(x_loc)
                    data_list.append(data)
    P = csr_matrix((data_list, (row_list, col_list)), shape=(num_path * N, num_OD * N))
    return P


# def save_p(N, spd_data, analysis_start_time, time_interval, single_date, path_list, OD_paths, new_file, incentive):
#     date_str = single_date.strftime("%Y-%m-%d")
#     print("date:", date_str)
#     P = get_P(N, spd_data, analysis_start_time, time_interval, single_date, path_list, OD_paths, incentive)
#     # print P
#     joblib.dump(P, os.path.join('P_matrix', new_file, date_str + '_' + str(incentive) + ".pickle"))
#

