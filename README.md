# Congestion Reduction via Personalized Incentives
With rapid population growth and urban development, traffic congestion has become an inescapable issue, especially in large cities. Many congestion reduction strategies have been proposed in the past, ranging from roadway extension to transportation demand management programs. In particular, congestion pricing schemes have been used as negative reinforcements for traffic control. In this project, we study an alternative approach of offering positive incentives to drivers to take alternative routes. More specifically, we propose an algorithm to reduce traffic congestion and improve routing efficiency via offering personalized incentives to drivers.  We exploit the wide-accessibility of smart communication devices to communicate with drivers and develop a look-ahead incentive offering mechanism using individuals’ routing and aggregate traffic information. The incentives are offered after solving a large-scale optimization problem in order to minimize the total travel time (or minimize any utility of the network such as total Carbon emission). Since this massive size optimization problem needs to be solved continually in the network, we developed a distributed computational approach where a major computational burden is carried out on the individual drivers' smart phones (and in parallel among drivers). We  theoretically show the convergence of the proposed distributed algorithm under a mild set of assumptions that are verified with real data.
# Synthetic data
Due to not being able to share our data with public, we ran createSyntheticData.py to generate synthetic data so you can follow the next steps to run the codes.

# Optimization model
### Requirements
- Python 3.6.8
- PyTorch 1.1.0
- Numpy 1.19.4+mkl
- Scipy 0.19.1
- NetworkX 1.11
- pickle
- joblib 0.14.0
- pandas 1.1.4
- PyYAML 5.1.2
- Matplotlib 2.2.3
- gurobipy 9.0.2
- psutil 5.7.2

## Learning graph of the network
First, define the desired setting on the file "optimization/YAML_graph/region_z.yaml".
Go to the 'optimization' folder. Run the following command:
```bash
python All_in_one.py --config_filename=YAML_graph/region_z.yaml
```

## Learning parameters of the BPR function
First, define the desired setting on the file "optimization/YAML_BPR/region_z.yaml".
Go to the 'optimization' folder. Run the following command to learn the parameters of the BPR function:
```bash
python link_capacity.py --config_filename=YAML_BPR/region_z.yaml
```
The extracted information for region z is provided in: 'optimizaiton/data/capcacity/region_z'

## Learning free flow travel time of routes
First, define the desired setting on the file "optimization/YAML_BPR/region_z.yaml".
Go to the 'optimization' folder. Run the following command:
```bash
python tt_BPR.py --config_filename=YAML_BPR/region_z.yaml
```


## Linear model (no incentive or baseline)
To solve these linear optimization model, first you should Follow the steps of OD estimation section to retrieve all matrices P, R, and Q. Next, you should copy and paste the latest created files from the following folders to the folders with same name in "optimization/DPFE_files":
1. observe_index_N
2. P_matrix
3. P_matrix_opt
4. Q_vector
5. R_matrix
6. tt
7. X_vector
Then, you need to specify the name of the generated folders in these 7 folders in YAML/region_z.yaml file as new_file:2021_03_28_03_35_11 (2021_03_28_03_35_11 is the name of my folder)

## Incentivizing - Linear model 
### Run model (data: region_z)
First, define the desired setting on the file "optimizaiton/YAML/region_z.yaml".
```bash
start_hour_opt: 7 # 7 AM
ub_multiplier_list: [4.] # Multiplier of upper bound of capacity constraints
incentive_list_available: [2, 10] # Incentives included in the data
incentive_list_temp: [2, 10]  # Incentives under analysis
new_file_temp: '2021_03_28_03_35_11' # [2, 10], temp=1
OD: ''  # Set of OD pairs
theta_temp: 1 
save_model_: True
region_: 'region_z'  
pad_str: '_pad'
n_hours: 7  # 3 AM - 10 AM
start_hour: 3  # 3 AM
start_minute_opt: 0  # 0 minutes
n_hr_test_horizon: 1  # 1 hour of analysis
interval_t: 15  # 15-minute intervals
analysis_month: 5
analysis_day: 1
num_paths_temp: 6
folder_DPFE: 'DPFE_files'
MIPGap_: 0.01  # Accuracy of the Gurobi solver
tot_budget_: [100000., 10000., 1000., 100., 0.]
plot_: True
one_arrival: False
```
Go to the 'optimization' folder. Next, run the following command:
```bash
python OptModel_linear.py --config_filename=YAML/region_z.yaml
```


# OD estimation
## Code
Code is an adapted and modified version of code of https://github.com/Lemma1/DPFE to our project.
### Requirements
- Python 2.7.18
- PyTorch*
- Numpy 1.13.3
- Scipy 0.19.1
- NetworkX 1.11
- pickle
- joblib 0.11
- pandas 0.18.1
- PyYAML 5.3.1
- Matplotlib 2.2.3

\*Find PyTorch for Python 2 on <a href="https://download.pytorch.org/whl/cu92/torch_stable.html" target="_top">link</a> and download "torch-1.5.0+cu92-cp27-cp27mu-linux_x86_64.whl".

## Learning graph of the network
First, define the desired setting on the file "OD_matrix_py27/YAML_graph/region_z.yaml".
Go to the 'OD_matrix_py27' folder. Run the following command:
```bash
python All_in_one.py --config_filename=YAML_graph/region_z.yaml
```

### Run the model (data: region_z)
First, define the desired setting on the file "OD_matrix_py27/YAML/region_z.yaml".
```bash
region_: region_z         # region of data (Do not change)
# new_file:  '??'  # Name of the yaml file if setting has been learnt before
new_file: null                      # Name of the yaml file if setting is new
interval_t: 15                      # Time interval in data (minutes)
n_hr: 7                             # Number of hours of data
AM_PM: False                        # If hours in data have AM-PM setting, then True
start_hr_0: 5  # 5 AM               # Starting hour of data without padding
finish_hr: 10  # 10 AM              # Finishing hour of data
pad_hr: 2                           # Number of padding hours in the beggining of data
num_paths: 6                        # Maximum number of paths from origin to destination
incentive_list: [5, 15]             # Set of incentives
theta_temp: 3                       # Temperature of Softmax function in computing choice probability
n_epochs: 20000                     # Number of epochs in training Neural Network for OD estimation
batch_size: 8192                    # Batch size in training Neural Network for OD estimation
lr: 0.01                            # Learning rate in training Neural Network for OD estimation
adagrad_: True                      # Using adagrad in training Neural Network for OD estimation
use_GPU_: True                      # Using GPU in training Neural Network for OD estimation
month_dict: {5: [1]}                # Days for which we do computatoin
OD: ''                         # Set of OD pairs ('' if we want all nodes to be OD points and '_sub1' if ODs in the sub1 set)
```
Go to the 'OD_matrix_py27' folder. Run the following command:
```bash
python run.py --config_filename=YAML/region_z.yaml
```
The generated files of matrix P, R, and travel time (based on historical data) and vector of OD estimation (q) are in the corresponding filesOD_add.


## Data
### Speed data
Speed data of the USC neighborhood in Los Angeles. The source of data is available data on ADMS system of USC. Data includes business days of March to May of 2018 from 3 to 10 AM. The speed data of 3 to 5 AM is added artificially and is equal to real speed at 5 AM. We call data of 3 AM to 5 AM a padded data.
Due to not being able to share the data with public, a synthetic data has been created.
Address of data: 'OD_matrix_py27/data/speed_volume/Mar2May_2018_region_z_AVG15_5-10_with_linkID_pad/my_link_avg_count_data_AVG15min_5-10_region_z_pad.pickle'
Foramt: The speed data is a pickled dictionary of 'pandas.DataFrame' files. Keys of the dictionary are link IDs. Indecies of the each DataFrame includes the date of the record and headers contain time of it. Here is an example of first three rows of DataFrame of link 0:
|            |    03:00:00   |   03:15:00   |    ...   |    09:45:00   |
|:----------:|:-------------:|:------------:|:--------:|:-------------:|
| 2018/03/01 |    19.48615   |   19.48615   |    ...   |    29.63890   | 
| 2018/03/02 |    20.25000   |   20.25000   |    ...   |    27.44445   |
| 2018/03/05 |    19.50000   |   19.50000   |    ...   |    30.62500   | 
|     ...    |      ...      |      ...     |    ...   |      ...      | 


### Volume data
Volume data of the USC neighborhood in Los Angeles. The source of data is available data on ADMS system of USC. Data includes business days of March to May of 2018 from 3 to 10 AM. The speed data of 3 to 5 AM is added artificially and is increasing gradually from 10% of real volume of 5 AM for 3 AM to real speed at 5 AM for 5 AM. We call data of 3 AM to 5 AM a padded data.
Due to not being able to share the data with public, a synthetic data has been created.
Address of data: 'OD_matrix_py27/data/speed_volume/Mar2May_2018_region_z_AVG15_5-10_with_linkID_pad/my_link_avg_spd_data_AVG15min_5-10_region_z_pad.pickle'
Foramt: The volume data is a pickled dictionary of 'pandas.DataFrame' files. Keys of the dictionary are link IDs. Indecies of the each DataFrame includes the date of the record and headers contain time of it. Here is an example of first three rows of DataFrame of link 0:
|            |    03:00:00   |   03:15:00   |    ...   |    09:45:00   |
|:----------:|:-------------:|:------------:|:--------:|:-------------:|
| 2018/03/01 |       1.0     |      3.0     |    ...   |      31.0     | 
| 2018/03/02 |       1.0     |      3.0     |    ...   |      31.0     |
| 2018/03/05 |       1.0     |      3.0     |    ...   |      31.0     | 
|     ...    |       ...     |      ...     |    ...   |       ...     | 

### OD points
List of origin points and destination points.
Address: 'OD_matrix_py27/data/graph/region_z/my_od_list_region_z_original.pickle'

### Graph of the network
Transportaion network in Networkx graph format.
Address: 'OD_matrix_py27/data/graph/region_z/my_graph_region_z_original.gpickle'

### YAML file (config)
File of the config of the OD estimation.
Address: 'OD_matrix_py27/YAML/region_z.yaml'
```bash
---
region_: region_z         # region of data (Do not change)
# new_file:  '??'  # Name of the yaml file if setting has been learnt before
new_file: null                      # Name of the yaml file if setting is new
interval_t: 15                      # Time interval in data (minutes)
n_hr: 7                             # Number of hours of data
AM_PM: False                        # If hours in data have AM-PM setting, then True
start_hr_0: 5  # 5 AM               # Starting hour of data without padding
finish_hr: 10  # 10 AM              # Finishing hour of data
pad_hr: 2                           # Number of padding hours in the beggining of data
num_paths: 6                        # Maximum number of paths from origin to destination
incentive_list: [2, 10]             # Set of incentives
theta_temp: 10                      # Temperature of Softmax function in computing choice probability
n_epochs: 10000                     # Number of epochs in training Neural Network for OD estimation
batch_size: 8192                    # Batch size in training Neural Network for OD estimation
lr: 0.01                            # Learning rate in training Neural Network for OD estimation
adagrad_: True                      # Using adagrad in training Neural Network for OD estimation
use_GPU_: True                      # Using GPU in training Neural Network for OD estimation
month_dict: {5: [1]}                # Days for which we do computatoin
OD: ''                         # Set of OD pairs ('' if we want all nodes to be OD points)
```

### Probability choice matrix
Matrix of choice probability of the driver given an incentive for a specific route at a specific time. The format is pickled dense Numpy array.
Address: 'OD_matrix_py27/P_matrix_opt'

### Route assignment matrix
Matrix of probability that a driver is at a link at a specific time. The format is pickled dense Numpy array.
Address: 'OD_matrix_py27/R_matrix'

### OD estimation
Estimation of number of drivers traveling between OD pairs in different times. The format is array.
Address: 'OD_matrix_py27/Q_matrix'

## Citation

If you find this repository useful in your research, please cite the following paper:
```
@article{ghafelebashi2023congestion,
  title={Congestion reduction via personalized incentives},
  author={Ghafelebashi, Ali and Razaviyayn, Meisam and Dessouky, Maged},
  journal={Transportation Research Part C: Emerging Technologies},
  volume={152},
  pages={104153},
  year={2023},
  publisher={Elsevier}
}
```
