import time
from copy import deepcopy
import numpy as np
import datetime
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import pickle
from collections import OrderedDict
import joblib
from base import *
from joblib import Parallel, delayed
import data_loader
from pfe import *  # !! My code
import argparse
import yaml


def AM_PM_f(t):
    if t > 12:
        return str(t - 12) + 'PM'
    elif t == 12:
        return '12PM'
    else:
        return str(t) + 'AM'


def hr_str(start_hr, finish_hr, AM_PM):
    if AM_PM:
        return AM_PM_f(start_hr), AM_PM_f(finish_hr)
    else:
        return str(start_hr), str(finish_hr)


def check_exists(add_, create_=True):
    if not os.path.exists(add_):
        print('Does not exist: \n' + add_)
        if create_:
            print('Creating the folder...')
            os.mkdir(add_)

def createFolder(address):
    if not os.path.exists(address):
	print('Creating folder ' + address + '')
	os.mkdir(address)
    else:
	print('Folder '+ address+' exists')

def main(args):
    # config_filename = 'YAML/region_x4_modified.yaml'
    config_filename = 'YAML/region_y2.yaml'
    with open(args.config_filename, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    region_ = config_dict['region_']
    interval_t = config_dict['interval_t']
    n_hr = config_dict['n_hr']
    AM_PM = config_dict['AM_PM']
    start_hr_0 = config_dict['start_hr_0']
    finish_hr = config_dict['finish_hr']
    pad_hr = config_dict['pad_hr']
    num_paths = config_dict['num_paths']
    incentive_list = config_dict['incentive_list']
    theta_temp = config_dict['theta_temp']
    n_epochs = config_dict['n_epochs']
    batch_size = config_dict['batch_size']
    lr = config_dict['lr']
    adagrad_ = config_dict['adagrad_']
    use_GPU_ = config_dict['use_GPU_']
    month_dict = config_dict['month_dict']
    OD_add = config_dict['OD']
    new_file = config_dict['new_file']


    col_width = max(len(word) for word in config_dict.keys()) + 4  # padding
    for a, b in config_dict.items():
        print "".join([a.ljust(col_width), str(b)])

  
    theta_OD_Estimation_temp = theta_temp
    theta_opt_temp = theta_temp
    if pad_hr:
        pad_str = '_pad'
        start_hr = start_hr_0 - pad_hr  # 5 AM
    else:
        pad_str = ''
        start_hr = start_hr_0

    analysis_start_time = datetime.time(start_hr, 0, 0)  # !! Starting time = 3 AM # !!
    time_basis = datetime.time(start_hr, 0, 0)  # !! 3 AM
    time_interval = datetime.timedelta(minutes=interval_t)  # !! Time interval = 15 minutes # !!
    n_times_per_hr = int(60 / interval_t)
    N = int(60 / interval_t * n_hr)  # !!

    start_hr_str, finish_hr_str = hr_str(start_hr_0, finish_hr, AM_PM)

    ##### Create ROW indecies of dataframe in Pandas
    # Number of days for each ID
    num_days = sum([len(x) for x in month_dict.itervalues()])
    date_need_to_finish = list()
    for iter_month in month_dict.keys():
        for iter_day in month_dict[iter_month]:
            # print '\n'
            date_temp = datetime.date(2018, iter_month, iter_day)  # !!
            # time_basis = datetime.time(3, 0, 0) # !! 3 AM
            cur_date_time = datetime.datetime.combine(date_temp, time_basis)
            # print cur_date_time
            single_date = cur_date_time.date()
            # print single_date
            date_need_to_finish.append(single_date)


    count_data_address = 'data/speed_volume/Mar2May_2018_' + region_ + '_AVG' + str(
        interval_t) + '_' + start_hr_str + '-' + finish_hr_str + '_with_linkID_pad/my_link_avg_count_data_AVG' + str(
        interval_t) + 'min_' + start_hr_str + '-' + finish_hr_str + '_' + region_ + pad_str + '.pickle'
    spd_data_address = 'data/speed_volume/Mar2May_2018_' + region_ + '_AVG' + str(
        interval_t) + '_' + start_hr_str + '-' + finish_hr_str + '_with_linkID_pad/my_link_avg_spd_data_AVG' + str(
        interval_t) + 'min_' + start_hr_str + '-' + finish_hr_str + '_' + region_ + pad_str + '.pickle'
    od_list_address = 'data/graph/' + region_ + '/my_od_list_' + region_ + '_original' + OD_add + '.pickle'
    graph_address = 'data/graph/' + region_ + '/my_graph_' + region_ + '_original.gpickle'

    # with open(count_data_address, 'rb') as handle:  # !!
    #     count_data = pickle.load(handle)
    # with open(spd_data_address, 'rb') as handle:  # !!
    #     spd_data = pickle.load(handle)

    count_data = pd.read_pickle(count_data_address)
    spd_data = pd.read_pickle(spd_data_address)

    print 'First 5 rows of count_data of link ' + str(count_data.keys()[0]) + ':', count_data[0].head()
    print 'First 5 rows of spd_data of link ' + str(spd_data.keys()[0]) + ':', spd_data[0].head()

    ### Read graph data
    with open(od_list_address, 'rb') as handle:  # !!
        (O_list, D_list) = pickle.load(handle)
    O_list = list(np.array(O_list).astype(int))
    D_list = list(np.array(D_list).astype(int))

    G = nx.read_gpickle(graph_address)  # !!
    G = nx.freeze(G)

    G2 = nx.read_gpickle(graph_address)  # !!

    ## Enumerate all paths
    if new_file==None:
        new_file = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        createFolder('X_vector')
        createFolder("observe_index_N")
        createFolder("R_matrix")
        createFolder("P_matrix")
        createFolder("P_matrix_opt")
        createFolder("plot")
        createFolder("Q_vector")
        createFolder("loss_vector")
        createFolder(os.path.join('Q_vector', new_file))
        createFolder('YAML')
        createFolder('tt')
        createFolder(os.path.join('X_vector', new_file))
        createFolder(os.path.join("observe_index_N", new_file))
        createFolder(os.path.join("R_matrix", new_file))
        createFolder(os.path.join("P_matrix", new_file))
        createFolder(os.path.join("P_matrix_opt", new_file))
        createFolder(os.path.join("plot", new_file))
        createFolder(os.path.join("Q_vector", new_file))
        createFolder(os.path.join("loss_vector", new_file))
        createFolder(os.path.join('Q_vector', new_file, 'python3'))
        createFolder(os.path.join('YAML', new_file))
        createFolder(os.path.join('tt', new_file))

    yaml_add = os.path.join('YAML', new_file, region_ + '_' + new_file + '.yaml')
    with open(yaml_add, 'w') as file:
        documents = yaml.dump(config_dict, file)

    start_time = time.time()
    OD_paths = OrderedDict()
    # Information of all links that are traversed, set of all link classes, keys are link ids
    link_dict = OrderedDict()
    # Set of all of the Path objects, one object for each path between OD pairs
    # OD_paths also include all the paths but with specifying the OD pair of the path but in path_list all path objects are together
    path_list = list()
    for O in O_list:
        for D in D_list:
            G_temp = deepcopy(G2)
            OD_temp = [O, D]
            paths = list()
            for iter_path in range(num_paths):
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
                    l_remove_nodes = [path_temp[iter] for iter in range(len(path_temp)) if bool_idx[iter] == True]
                    if l_remove_nodes:
                        for iter_remove in range(len(l_remove_nodes)):
                            G_temp.remove_node(l_remove_nodes[iter_remove])

                except nx.NetworkXNoPath:
                    # print 'No more path between ' + str(O) + ' and ' + str(D)
                    break
            # print 'paths:', paths
            print "From ", O, " To ", D, "there is/are ", len(paths), "path(s)"

            # If the number of paths between O and D is at least 1
            if len(paths) != 0:
                # We create tmp_path_list and fill it with the path objects in the 'base.py' code
                # Next we add all these path objects for O and D to OD_paths[(O, D)]
                tmp_path_list = list()
                for path in paths:
                    # path_o is a Path object
                    path_o = Path();
                    # this path is now an attribute of Path object
                    path_o.node_list = path;
                    # Constructs the
                    path_o.node_to_list(G, link_dict);

                    tmp_path_list.append(path_o);
                    path_list.append(path_o);
                # Add the list of all Path classes for the pair O & D, Each Path class includes all the attributes
                OD_paths[(O, D)] = tmp_path_list;
            # counter += 1

    OD_paths_opt = OrderedDict()
    link_dict_opt = OrderedDict()
    path_list_opt = list()
    for O in O_list:
        for D in D_list:
            G_temp = deepcopy(G2)
            OD_temp = [O, D]
            paths_opt = list()
            for iter_path in range(num_paths):
                try:
                    path_temp_opt = list(k_shortest_paths(G_temp, O, D, 1))[0]  # !! My code
                    paths_opt.append(path_temp_opt)
                    if len(path_temp_opt) <= 2:
                        break
                    bool_idx = [path_temp_opt[iter] not in OD_temp for iter in range(len(path_temp_opt))]
                    l_remove_nodes = [path_temp_opt[iter] for iter in range(len(path_temp_opt)) if bool_idx[iter] == True]
                    if l_remove_nodes:
                        for iter_remove in range(len(l_remove_nodes)):
                            G_temp.remove_node(l_remove_nodes[iter_remove])
                except nx.NetworkXNoPath:
                    # print 'No more path between ' + str(O) + ' and ' + str(D)
                    break
                    # print 'paths:', paths
            # print "From ", O, " To ", D, "there is/are ", len(paths_opt), "path(s)"
            if len(paths_opt) != 0:
                tmp_path_list_opt = list()
                for path_opt in paths_opt:
                    path_o_opt = Path();
                    path_o_opt.node_list = path_opt;
                    path_o_opt.node_to_list(G2, link_dict);
                    tmp_path_list_opt.append(path_o_opt);
                    path_list_opt.append(path_o_opt);
                OD_paths_opt[(O, D)] = tmp_path_list_opt;
    print "Generating paths in %.2f minutes." % ((time.time() - start_time) / 60)


    ## Generate Delta
    # Number of OD pairs
    num_OD = len(OD_paths)
    print "Number of OD pairs: ", num_OD
    link_list = list(link_dict.values())
    num_link = len(link_list)
    print "Number of links: ", num_link
    # .itervalues(): returns an iterator over the values of dictionary dictionary
    # vector of number of paths between each OD pair
    num_path_v = [len(x) for x in OD_paths.itervalues()]
    fig = plt.figure()
    plt.hist(num_path_v, edgecolor='black', linewidth=1.2)
    plt.xlabel('Number of paths')
    plt.ylabel('Number of OD pairs')
    plt.title('Histogram of number of paths of OD pairs')
    #    plt.show()
    fig.savefig("plot/" + new_file + "/hist_num_paths.png")


    # Total number of paths
    num_path = np.sum(num_path_v)
    print "Number of paths: ", num_path
    assert (len(path_list) == num_path)


    # The delta matrix with bianry elements
    delta = np.zeros((num_link, num_path))
    # Iterate over links (edges)
    for i, link in enumerate(link_list):
        # Iterate over paths
        for j, path in enumerate(path_list):
            # If the path includes the link (edge), we change the element of the matrix to 1
            if link in path.link_list:
                delta[i, j] = 1.0


    link_loc = dict()
    for idx, link in enumerate(link_list):
        # print link.ID
        link_loc[link] = idx


    # ############################################################################
    ## Construct R matrix
    # A parallel computing framework is used to compute the R matrix as well as P matrix. Since we have a 6 core CPU, so we use 5 process to run the program, leaving one core to ensure the desktop does not get stuck.
    # tmp_date is iterating over all the possible days >> for tmp_date in date_need_to_finish) >> between (2014, 1, 1) and (2016, 12, 31)
    start_time = time.time()
    Parallel(n_jobs=-1, temp_folder='temp', max_nbytes='10M')(
        delayed(save_r)(N, spd_data, analysis_start_time, time_interval,
                        tmp_date, link_dict, link_list, link_loc, path_list, new_file) for tmp_date in date_need_to_finish)
    t_ = (time.time() - start_time) / 60
    print "save_r: %.2f minutes" % t_

    # ############################################################################
    ## Construct P matrix
    #### parallel computing
    start_time = time.time()
    theta_OD_Estimation = theta_OD_Estimation_temp  # !!!!!!!
    Parallel(n_jobs=-1)(delayed(save_p)(N, spd_data, analysis_start_time, time_interval,
                                        tmp_date, path_list, OD_paths, new_file, theta_OD_Estimation) for tmp_date in
                        date_need_to_finish)
    t_ = (time.time() - start_time) / 60
    print "save_p: %.2f minutes" % t_

    start_time = time.time()
    theta_opt = theta_opt_temp  # !!!!!!!
    Parallel(n_jobs=-1)(delayed(save_p_opt0)(N, spd_data, analysis_start_time, time_interval,
                                             tmp_date, path_list_opt, OD_paths_opt, new_file, theta_opt)
                        for tmp_date in date_need_to_finish)
    t_ = (time.time() - start_time) / 60
    print "save_p_opt0: %.2f minutes" % t_

    start_time = time.time()
    theta_opt = theta_opt_temp  # !!!!!!!
    Parallel(n_jobs=-1)(delayed(save_p_opt)(N, spd_data, analysis_start_time, time_interval,
                                            tmp_date, path_list_opt, OD_paths_opt, new_file, incentive, theta_opt) for
                        incentive in incentive_list
                        for tmp_date in date_need_to_finish)
    t_ = (time.time() - start_time) / 60
    print "save_p_opt: %.2f minutes" % t_
    # ############################################################################
    ## Construct travel time matrix
    start_time = time.time()
    theta_opt = theta_opt_temp  # !!!!!!!
    for tmp_date in date_need_to_finish:
        save_tt(N, spd_data, analysis_start_time, time_interval, tmp_date, path_list_opt, OD_paths_opt, new_file)
    Parallel(n_jobs=-1)(delayed(save_tt_joblib)(N, spd_data, analysis_start_time, time_interval,
                                            tmp_date, path_list_opt, OD_paths_opt, new_file)
                        for tmp_date in date_need_to_finish)
    t_ = (time.time() - start_time) / 60
    print "save_p_opt: %.2f minutes" % t_

    # ############################################################################

    ## Construct link flow vector
    # The filter() method filters the given sequence with the help of a function that tests each element in the sequence to be true or not.
    # o_link_list is the list of observed links. In other words, the links that their cound_data is available
    o_link_list = filter(lambda x: x.ID in count_data.keys(), link_list)  # !!

    def get_x_o(N, o_link_list, tmp_date, analysis_start_time, time_interval, count_data):
        # Number of observed links
        num_o_link = len(o_link_list)
        # Vector "x" contains the volume of all links for a specific day
        x = np.zeros(num_o_link * N)
        for h in xrange(N):
            start_time = (datetime.datetime.combine(tmp_date, analysis_start_time) + h * time_interval).time()
            for a, link in enumerate(o_link_list):
                data = np.float(count_data[link.ID].loc[tmp_date][start_time])
                x[h * num_o_link + a] = data
        return x

    # X vector
    start_time = time.time()
    for iter_month in month_dict.keys():
        for iter_day in month_dict[iter_month]:
            date_temp = datetime.date(2018, iter_month, iter_day)  # !!
            cur_date_time = datetime.datetime.combine(date_temp, time_basis)
            single_date = cur_date_time.date()
            date_str = single_date.strftime("%Y-%m-%d")
            x = get_x_o(N, o_link_list, single_date, analysis_start_time, time_interval, count_data)
            np.save(os.path.join('X_vector', new_file, date_str), x)

    t_ = (time.time() - start_time) / 60
    print "X_vector: %.2f minutes" % t_


    ## Create the observed delta (time dependent)
    observe_index = np.array(map(lambda x: x in o_link_list, link_list)).astype(np.int)
    observe_index_N = np.tile(observe_index, (N,))
    np.save(os.path.join("observe_index_N", new_file, "observe_index_N"), observe_index_N)


    ## Batch computing for all dates
    # This sessesion is used to run the DODE method for ALL DATES


    def get_qest2(tmp_date, P_date_dict):

        date_str = tmp_date.strftime("%Y-%m-%d")
        P_date_str = P_date_dict[tmp_date].strftime("%Y-%m-%d")
        observe_index_N = np.load(os.path.join("observe_index_N", new_file, "observe_index_N.npy"))

        x_o = np.load(os.path.join('X_vector', new_file, date_str + ".npy"))
        r = joblib.load(os.path.join("R_matrix", new_file, date_str + ".pickle")).tocsr()
        P = joblib.load(os.path.join("P_matrix", new_file, date_str + "_0.pickle")).tocsr()
        A = np.array(r.dot(P).todense())[observe_index_N == 1, :]
        (q_est, r_norm, list_loss) = nnls(A, x_o, n_epochs, batch_size, lr, adagrad=adagrad_, use_GPU=use_GPU_)  # !! My code
        print "\n", date_str, r_norm
        pickle.dump((q_est, r_norm), open(os.path.join('Q_vector', new_file, date_str + '.pickle'), 'wb'))
        pickle.dump((list_loss), open(os.path.join('loss_vector', new_file, date_str + '.pickle'), 'wb'))

        x_est = A.dot(q_est)  # !! My code
        fig = plt.figure()
        plt.plot(x_o, x_est, 'o', markersize=1.5)  # !! My code
        plt.plot(x_o, x_o, 'r')  # !! My code
        # plt.show() # !! My code
        fig.savefig("plot/" + new_file + "/" + date_str + ".png")

        perc_q1 = 100 * (float(q_est[q_est >= 0].shape[0]) / float(q_est.shape[0]))
        perc_q2 = 100 * (float(q_est[q_est == 0].shape[0]) / float(q_est.shape[0]))
        print "Percentage of non-negative values in the q_est: %.2f" % perc_q1
        print "Percentage of zero values in the q_est: %.2f" % perc_q2

        fig2 = plt.figure()
        # plt.figure(figsize=(12, 8))
        plt.rc('xtick', labelsize=16)  # !! My code
        plt.rc('ytick', labelsize=16)  # !! My code
        plt.title("Loss plot, " + date_str, size=16)  # !! My code
        plt.xlabel('Iteration', size=16)  # !! My code
        plt.ylabel('Error', size=16)  # !! My code
        plt.plot(list_loss)  # !! My code
        # plt.show()  # !! My code
        fig2.savefig("plot/" + new_file + "/loss_"+ date_str + ".png")

        if len(list_loss) >= 2000:
            fig3 = plt.figure()
            plt.figure(figsize=(12, 8))
            plt.rc('xtick', labelsize=16)  # !! My code
            plt.rc('ytick', labelsize=16)  # !! My code
            plt.title("Loss plot without first 2000 loss values, " + date_str, size=16)  # !! My code
            plt.xlabel('Iteration', size=16)  # !! My code
            plt.ylabel('Error', size=16)  # !! My code
            plt.plot(list_loss[2000:])  # !! My code
            # plt.show()  # !! My code
            fig3.savefig("plot/"+new_file+"/loss_WOFirst2000_"+date_str+".png")


    P_date_dict = dict()
    for iter_month in month_dict.keys():  # !! My code
        for iter_day in month_dict[iter_month]:  # !! My code
            date_temp = datetime.date(2018, iter_month, iter_day)  # !!
            cur_date_time = datetime.datetime.combine(date_temp, time_basis)
            single_date = cur_date_time.date()
            # Here, we assign the closest date to single date. As we are using each day for itself so P_date_dict[single_date] = single_date
            P_date_dict[single_date] = single_date


    for iter_month in month_dict.keys():
        for iter_day in month_dict[iter_month]:
            start_time = time.time()
            # print '\n'
            date_temp = datetime.date(2018, iter_month, iter_day)  # !!
            cur_date_time = datetime.datetime.combine(date_temp, time_basis)
            single_date = cur_date_time.date()
            date_str = single_date.strftime("%Y-%m-%d")
            print date_str
            # get_qest2 will check P_date_dict to use the probablity matrix of the closest day to single_date
            get_qest2(single_date, P_date_dict)
            print "Date: %s, Run time: %.2f minutes" % (date_temp, (time.time() - start_time) / 60)


    def data2py3(date_temp):
        date_str = os.path.join('Q_vector', new_file, date_temp.strftime("%Y-%m-%d") + '.pickle')
        print date_str
        with open(date_str, 'rb') as f:
            x = pickle.load(f)
        date_str2 = os.path.join('Q_vector', new_file, 'python3', date_temp.strftime("%Y-%m-%d") + '.pickle')
        print date_str2
        data_loader.data_pickle(list(x[0]), date_str2)


    for iter_month in month_dict.keys():
        for iter_day in month_dict[iter_month]:
            date_temp = datetime.date(2018, iter_month, iter_day)
            data2py3(date_temp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename',
                        # default='YAML/region_x4_modified.yaml',
                        default='YAML/region_z.yaml',
                        type=str,
                        help='Configuration filename for the region.')
    args = parser.parse_args()
    main(args)
