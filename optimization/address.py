import os


def pickle_f(pickle_file_):
    try:
        folder_ = 'data\\speed_volume'
        if pickle_file_ == 'may_2018_hist1_12_pred1_12_1PM_7PM_region_x1_sub1':
            name_ = 'mar2may_2018_hist1_12_pred1_12\\may_2018_hist1_12_pred1_12_1PM_7PM_region_x1_sub1.pkl'
        elif pickle_file_ == 'inf_0_60_hist60_NN_may_2018_1PM_7PM_region_x1_sub1_test':
            name_ = 'mar2may_2018_hist1_12_pred1_12\\inf_0_60_hist60_NN_may_2018_1PM_7PM_region_x1_sub1_test.pkl'
        elif pickle_file_ == 'may_2018_hist1_12_pred1_12_5AM_8AM_region_x1_sub1_OptModel':
            name_ = 'mar2may_2018_hist1_12_pred1_12_5AM_8AM_region_x1_sub1_OptModel\\data_0_60_hist60_OptModel_may_2018_5AM_8AM_region_x1_sub1.pkl'
        elif pickle_file_ == 'inf_0_60_hist60_NN_may_2018_5AM_8AM_region_x1_sub1_test_OptModel':
            name_ = 'mar2may_2018_hist1_12_pred1_12_5AM_8AM_region_x1_sub1_OptModel\\inf_0_60_hist60_OptModel_may_2018_5AM_8AM_region_x1_sub1.pkl'
        else:
            name_ = 0
        add_1 = os.path.join(folder_, name_)
    except FileNotFoundError:
        print('Pickled file of \'' + pickle_file_ + '\' does not exist.')
    return add_1


def not_pickle_f(not_pickle_file_):
    folder_ = 'data\\speed_volume'
    try:
        if not_pickle_file_ == 'may_2018_hist1_12_pred1_12_1PM_7PM_region_x1_sub1':
            # mar2apr X: 3 features, 12 times, features: 1.day_number 2.hr_min 3.speed
            csv_file_1 = 'mar2may_2018_hist1_12_pred1_12_1PM_7PM_region_x1_sub1\\X_0_60_hist60_NN_may_2018_1PM_7PM_region_x1_sub1_test.csv'
            # mar2apr Y: 1 features, 12 times, feature: 1.speed
            csv_file_2 = 'mar2may_2018_hist1_12_pred1_12_1PM_7PM_region_x1_sub1\\Y_0_60_hist60_NN_may_2018_1PM_7PM_region_x1_sub1_test.csv'
        elif not_pickle_file_ == 'inf_0_60_hist60_NN_may_2018_1PM_7PM_region_x1_sub1_test':
            csv_file_1 = 'mar2may_2018_hist1_12_pred1_12_1PM_7PM_region_x1_sub1\\inf_0_60_hist60_NN_may_2018_1PM_7PM_region_x1_sub1_test.csv'
            csv_file_2 = 'NONE'
        elif not_pickle_file_ == 'may_2018_hist1_12_pred1_12_5AM_8AM_region_x1_sub1_OptModel':
            csv_file_1 = 'mar2may_2018_hist1_12_pred1_12_5AM_8AM_region_x1_sub1_OptModel\\data_0_60_hist60_OptModel_may_2018_5AM_8AM_region_x1_sub1.csv'
            csv_file_2 = 'NONE'
        elif not_pickle_file_ == 'inf_0_60_hist60_NN_may_2018_5AM_8AM_region_x1_sub1_test_OptModel':
            csv_file_1 = 'mar2may_2018_hist1_12_pred1_12_5AM_8AM_region_x1_sub1_OptModel\\inf_0_60_hist60_OptModel_may_2018_5AM_8AM_region_x1_sub1.csv'
            csv_file_2 = 'NONE'

        add_1 = os.path.join(folder_, csv_file_1)
        add_2 = os.path.join(folder_, csv_file_2)
    except FileNotFoundError:
        print('File of \'' + not_pickle_file_ + '\' does not exist.')

    return add_1, add_2



def model_f(info, gpu_, folder_, pickle_file_):
    # N is batch size; D_in is input dimension (number of features);
    # H is hidden dimension; D_out is output dimension.
    # D_in, H1, H2, H3, D_out, N, alpha, t = info['D_in'], info['H1'], info['H2'], info['H3'], info['D_out'], info['N'], \
    #                                        info['alpha'], info['t'] # deprecated
    D_in, H1, H2, H3, D_out, N, test_n_epochs, data_name = info['D_in'], info['H1'], info['H2'], info['H3'], info['D_out'], \
                                               info['N'], info['test_n_epochs'], info['data_name']
    # 0 batch percentage means no minmax problem
    # batch_percentage = info['batch_percentage'] # deprecated
    MSE_ = info['MSE_']
    # Multiplier of the regularizer
    w_regul = info['w_regul']
    regul = '_L2_regul_'

    if MSE_:
        loss_name = 'MSE'
    else:
        loss_name = 'L1'

    if gpu_:
        processor = '_gpu_'
    else:
        processor = '_cpu_'
    # name_1 = 'NN_minmax_metrla_70_20_hist1_12_pred1_12' + regul + str(w_regul) + '_Loss_Norm_' + loss_name\
    #          + '_max_' + str(batch_percentage) + '_N_' + str(N) + '_D_in_' + str(D_in) + '_alpha_' + str(alpha) + \
    #          '_t_' + str(t) + '_test_n_epochs_' + str(test_n_epochs) + processor # deprecated
    if data_name == 'simple_NN':
        name_1 = data_name + '_' + pickle_file_ + regul + str(w_regul) + '_Loss_Norm_' + loss_name\
             + '_N_' + str(N) + '_D_in_' + str(D_in) + '_test_n_epochs_' + str(test_n_epochs) + processor
    if H2 == 0:
        name_ = name_1 + '_H1_' + str(H1)
    elif H3 == 0:
        name_ = name_1 + '_H1_' + str(H1) + '_H2_' + str(H2)
    else:
        name_ = name_1 + '_H1_' + str(H1) + '_H2_' + str(H2) + '_H3_' + str(H3)

    add_ = os.path.join(folder_, name_)
    return add_


def model_capacity(pickle_file_, gpu_):
    if gpu_:
        processor = '_gpu_'
    else:
        processor = '_cpu_'

    name_ = pickle_file_ + processor

    folder_ = 'saved_models/model_capacity'
    add_ = os.path.join(folder_, name_)

    return(add_)
