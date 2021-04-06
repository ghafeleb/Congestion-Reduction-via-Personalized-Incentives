import numpy as np
import pandas as pd
import pickle
import sys
import argparse
import yaml
import os


def loadPickle(address):
	return pickle.load(open(address, 'rb'))

def writePickle(x, address):
	pickle.dump(x, open(address, 'wb'))


def createFolder(address):
    if not os.path.exists(address):
        print('Creating folder ' + address + '')
        os.mkdir(address)
    else:
        print(f'Folder \'{address}\' exists')


def my_ID_Creation(config_dict):
	'''
	Data files of link_capacity.py >> 
	'''
	region = config_dict['region']
	addOpt = config_dict['addOpt']
	addODEstimation = config_dict['addODEstimation']
	subAddData = config_dict['subAddData']
	subAddGraph = config_dict['subAddGraph']
	subAddRegion = config_dict['subAddRegion']

	addData = os.path.join(addOpt, subAddData)
	addGraph = os.path.join(addData, subAddGraph)
	addRegion = os.path.join(addGraph, subAddRegion)

	createFolder(addOpt)
	createFolder(addData)
	createFolder(addGraph)
	createFolder(addRegion)
	
	addODEstimationData = os.path.join(addODEstimation, subAddData)
	addODEstimationGraph = os.path.join(addODEstimationData, subAddGraph)
	addODEstimationRegion = os.path.join(addODEstimationGraph, subAddRegion)

	createFolder(addODEstimation)
	createFolder(addODEstimationData)
	createFolder(addODEstimationGraph)
	createFolder(addODEstimationRegion)

	# Create my_ID_region
	subAddMyID = 'my_ID_' + region + '_original'
	addMyID = os.path.join(addRegion, subAddMyID)
	addMyIDODEstimation = os.path.join(addODEstimationRegion, subAddMyID)
	print(f'Address of my_ID_region file: {addMyID}')
	myIDList = config_dict['myIDList']
	myIDDataFrame = pd.DataFrame(myIDList)
	myIDDataFrame.to_csv(addMyID + '.csv', index=False, header=False)
	myIDDataFrame.to_csv(addMyIDODEstimation + '.csv', index=False, header=False)
	writePickle(myIDDataFrame, addMyID + '.pickle')
	writePickle(myIDDataFrame, addMyIDODEstimation + '.pickle')


def data_speed_volume(config_dict):
	region = config_dict['region']
	addOpt = config_dict['addOpt']
	subAddData = config_dict['subAddData']
	subAddSpeedVolume = config_dict['subAddSpeedVolume']
	interval_t = config_dict['interval_t']
	start_hr = config_dict['start_hr']
	finish_hr = config_dict['finish_hr']
	data_year = config_dict['data_year']
	months_ = config_dict['months_']
	myIDList = config_dict['myIDList']
	n_days = config_dict['n_days']
	n_hour = finish_hr-start_hr

	addData = os.path.join(addOpt, subAddData)
	createFolder(addData)
	addSpeedVolume = os.path.join(addData, subAddSpeedVolume)
	createFolder(addSpeedVolume)
	
	subAddSpeedVolume2 = months_ + '_' + str(data_year) + '_AVG' + str(interval_t) + '_' + str(start_hr) + '-' + str(finish_hr) + '_' + region + '_with_linkID'
	addSpeedVolume2 = os.path.join(addSpeedVolume, subAddSpeedVolume2)
	createFolder(addSpeedVolume2)
		
	# Create speed data
	subAddSpeed = 'speed_' + months_ + '_' + str(data_year) + '_new_' +  str(start_hr) + '-' + str(finish_hr) + '_' + region
	addSpeed = os.path.join(addSpeedVolume2, subAddSpeed)
	print(f'Address of speed file: {addSpeed}')
	speedArrayTemp = np.ones((n_days*len(myIDList), n_hour*int(60/interval_t)+1)) * 70
	for iter in range(len(myIDList)):
		speedArrayTemp[(n_days*iter):(n_days*(iter+1)), 0] = myIDList[iter]
	speedDataFrame = pd.DataFrame(speedArrayTemp)
	speedDataFrame.to_csv(addSpeed + '.csv', index=False, header=False)
	# writePickle(myIDDataFrame, addMyID + '.pickle')
	
	# Create volume data
	subAddVolume = 'volume_' + months_ + '_' + str(data_year) + '_new_' +  str(start_hr) + '-' + str(finish_hr) + '_' + region
	addVolume = os.path.join(addSpeedVolume2, subAddVolume)
	print(f'Address of volume file: {addVolume}')
	volumeArrayTemp = np.ones((n_days*len(myIDList), n_hour*int(60/interval_t)+1)) * 30
	for iter in range(len(myIDList)):
		volumeArrayTemp[(n_days*iter):(n_days*(iter+1)), 0] = myIDList[iter]
	volumeDataFrame = pd.DataFrame(volumeArrayTemp)
	volumeDataFrame.to_csv(addVolume + '.csv', index=False, header=False)
	# writePickle(myIDDataFrame, addMyID + '.pickle')


	addODEstimation = config_dict['addODEstimation']
	interval_t_ODEstimation = config_dict['interval_t_ODEstimation']
	start_hr_ODEstimation = config_dict['start_hr_ODEstimation']
	finish_hr_ODEstimation = config_dict['finish_hr_ODEstimation']
	n_pad_hour = config_dict['n_pad_hour']
	n_hour_OD = finish_hr_ODEstimation-start_hr_ODEstimation + n_pad_hour
	addODEstimationData = os.path.join(addODEstimation, subAddData)
	addODEstimationSpeedVolume = os.path.join(addODEstimationData, subAddSpeedVolume)

	createFolder(addODEstimation)
	createFolder(addODEstimationData)
	createFolder(addODEstimationSpeedVolume)

	subAddSpeedVolume3 = months_ + '_' + str(data_year) + '_AVG' + str(interval_t_ODEstimation) + '_' + str(start_hr_ODEstimation) + '-' + str(finish_hr_ODEstimation) + '_' + region + '_with_linkID_pad'
	addSpeedVolume3 = os.path.join(addODEstimationSpeedVolume, subAddSpeedVolume3)
	createFolder(addSpeedVolume3)
	subAddSpeedVolume4 = months_ + '_' + str(data_year) + '_' + region + '_AVG' + str(interval_t_ODEstimation) + '_' + str(start_hr_ODEstimation) + '-' + str(finish_hr_ODEstimation) + '_with_linkID_pad'
	addSpeedVolume4 = os.path.join(addODEstimationSpeedVolume, subAddSpeedVolume4)
	createFolder(addSpeedVolume4)
	
	# Create speed data for OD estimation
	subAddSpeedOD = 'speed_' + months_ + '_' + str(data_year) + '_new_' +  str(start_hr_ODEstimation) + '-' + str(finish_hr_ODEstimation) + '_' + region + '_pad'
	addSpeedOD = os.path.join(addSpeedVolume3, subAddSpeedOD)
	addSpeedOD2 = os.path.join(addSpeedVolume4, subAddSpeedOD)
	print(f'Address of speed file of OD: {addSpeedOD}')
	speedArrayTemp2 = np.ones((n_days*len(myIDList), n_hour_OD*int(60/interval_t_ODEstimation)+1)) * 70
	for iter in range(len(myIDList)):
		speedArrayTemp2[(n_days*iter):(n_days*(iter+1)), 0] = myIDList[iter]
	n_times_pad = n_pad_hour*int(60/interval_t_ODEstimation)
	for iter in range(n_times_pad):
		speedArrayTemp2[:, iter+1] *= (iter/n_times_pad) 
	speedDataFrame2 = pd.DataFrame(speedArrayTemp2)
	print(f'addSpeedOD:{addSpeedOD}')
	speedDataFrame2.to_csv(addSpeedOD + '.csv', index=False, header=False)
	speedDataFrame2.to_csv(addSpeedOD2 + '.csv', index=False, header=False)
	# writePickle(myIDDataFrame, addMyID + '.pickle')
	
	# Create volume data for OD estimation
	subAddVolumeOD = 'volume_' + months_ + '_' + str(data_year) + '_new_' +  str(start_hr_ODEstimation) + '-' + str(finish_hr_ODEstimation) + '_' + region + '_pad'
	addVolumeOD = os.path.join(addSpeedVolume3, subAddVolumeOD)
	addVolumeOD2 = os.path.join(addSpeedVolume4, subAddVolumeOD)
	print(f'Address of volume file of OD: {addVolumeOD}')
	volumeArrayTemp2 = np.ones((n_days*len(myIDList), n_hour_OD*int(60/interval_t_ODEstimation)+1)) * 30
	for iter in range(len(myIDList)):
		volumeArrayTemp2[(n_days*iter):(n_days*(iter+1)), 0] = myIDList[iter]
	n_times_pad = n_pad_hour*int(60/interval_t_ODEstimation)
	for iter in range(n_times_pad):
		volumeArrayTemp2[:, iter+1] *= (iter/n_times_pad) 
	volumeDataFrame2 = pd.DataFrame(volumeArrayTemp2)
	volumeDataFrame2.to_csv(addVolumeOD + '.csv', index=False, header=False)
	volumeDataFrame2.to_csv(addVolumeOD2 + '.csv', index=False, header=False)
	# writePickle(myIDDataFrame, addMyID + '.pickle')



def plot_capacity(config_dict):
	addOpt = config_dict['addOpt']
	subAddPlot = config_dict['subAddPlot']
	subAddFitCurve = config_dict['subAddFitCurve']
	subAddLossFunction = config_dict['subAddLossFunction']
	subAddRegion = config_dict['subAddRegion']
	subAddFitData = config_dict['subAddFitData']

	addPlot = os.path.join(addOpt, subAddPlot)
	createFolder(addPlot)
	addFitCurve = os.path.join(addPlot, subAddFitCurve)
	createFolder(addFitCurve)
	addLossFunction = os.path.join(addFitCurve, subAddLossFunction)
	createFolder(addLossFunction)
	addPlotRegion = os.path.join(addLossFunction, subAddRegion)
	createFolder(addPlotRegion)
	addFitData = os.path.join(addFitCurve, subAddFitData)
	createFolder(addFitData)


	addODEstimation = config_dict['addODEstimation']
	createFolder(addODEstimation)
	addPlot2 = os.path.join(addODEstimation, subAddPlot)
	createFolder(addPlot2)


def adjacency_matrix(config_dict):
	region = config_dict['region']
	addOpt = config_dict['addOpt']
	subAddData = config_dict['subAddData']
	subAddGraph = config_dict['subAddGraph']
	subAddRegion = config_dict['subAddRegion']

	addData = os.path.join(addOpt, subAddData)
	addGraph = os.path.join(addData, subAddGraph)
	addRegion = os.path.join(addGraph, subAddRegion)

	createFolder(addOpt)
	createFolder(addData)
	createFolder(addGraph)
	createFolder(addRegion)
	
	# Inventory file
	data_year = config_dict['data_year']
	months_inventory = config_dict['months_inventory']
	myIDList = config_dict['myIDList']
	latitudeList = config_dict['latitudeList']
	longitudeList = config_dict['longitudeList']
	inventoryDataFrame = pd.DataFrame({'link_id': myIDList, 'latitude': latitudeList, 'longitude': longitudeList})
	subAddInventory = str(data_year) + '_' + months_inventory + '_location_table' 
	addInventory = os.path.join(addGraph, subAddInventory)
	inventoryDataFrame.to_csv(addInventory + '.csv', index=False, header=True)

	# Adjacency matrix file
	subAddAdj = 'AdjMatrix_' + region + '_original' 
	addAdj = os.path.join(addRegion, subAddAdj)
	AdjDataFrame = pd.DataFrame({'origin': myIDList, 'destination': myIDList[::-1], 'direction': [3, 2], 3: myIDList})
	AdjDataFrame.to_csv(addAdj + '.csv', index=False, header=True)

	# AdjMatrix_region_x4_modified_original.csv


	addODEstimation = config_dict['addODEstimation']
	addDataODEstimation = os.path.join(addODEstimation, subAddData)
	addGraphODEstimation = os.path.join(addDataODEstimation, subAddGraph)
	addRegionODEstimation = os.path.join(addGraphODEstimation, subAddRegion)

	createFolder(addODEstimation)
	createFolder(addDataODEstimation)
	createFolder(addGraphODEstimation)
	createFolder(addRegionODEstimation)
	
	addInventoryODEstimation = os.path.join(addODEstimation, subAddInventory)
	inventoryDataFrame.to_csv(addInventoryODEstimation + '.csv', index=False, header=True)
	addInventoryODEstimation2 = os.path.join(addGraphODEstimation, subAddInventory)
	inventoryDataFrame.to_csv(addInventoryODEstimation2 + '.csv', index=False, header=True)
	addAdjODEstimation = os.path.join(addRegionODEstimation, subAddAdj)
	AdjDataFrame = pd.DataFrame({'origin': myIDList, 'destination': myIDList[::-1], 'direction': [3, 2], 3: myIDList})
	AdjDataFrame.to_csv(addAdjODEstimation + '.csv', index=False, header=True)




def main(args):
	with open(args.config_filename, 'r') as f:
		config_dict = yaml.load(f, Loader=yaml.FullLoader)

	my_ID_Creation(config_dict)
	data_speed_volume(config_dict)
	plot_capacity(config_dict)
	adjacency_matrix(config_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename',
                        default='region_z.yaml',
                        type=str,
                        help='Configuration filename for the region.')
    args = parser.parse_args()
    main(args)
