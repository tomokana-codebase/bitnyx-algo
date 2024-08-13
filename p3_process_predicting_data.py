
import time
import pandas as pd
import numpy as np
import os
import requests
import random
from pandas import json_normalize
# from 4_predict_with_lstm import predict_with_lstm

from termcolor import colored


# HOME_DIR = os.getcwd()+'/bots/models/models/algos/orthos/'

    
def process_predicting_data(config_id, bot_id, HOME_DIR):
    api_fetch_predicting_data(config_id, HOME_DIR)
    pad_and_sort_predicting_data(config_id, HOME_DIR)
    label_data_targets_and_inputs(config_id, HOME_DIR)
    normalize_predicting_data(config_id, HOME_DIR)
    sequence_predicting_data(config_id, HOME_DIR)


def api_fetch_predicting_data(config_id, HOME_DIR):

    output_data_file = os.path.abspath(os.path.join(HOME_DIR, "config{}_files/predicting/0_api_predicting_data.csv".format(config_id)))
    config = pd.read_csv(os.path.abspath(os.path.join(HOME_DIR, 'configs.csv')), index_col='config_id').loc[config_id]

    # while(True):
    no_of_datapoints = config['sequence_size']+config['horizon']+1
    data_frequency = config['data_frequency']
    # print ("no_of_datapoints: {}".format(no_of_datapoints))

    # data = get_glassnode_data(no_of_datapoints, config['data_frequency'])
    file_id = len(os.listdir(os.path.abspath(os.path.join(HOME_DIR, 'api_data_{}/'.format(data_frequency)))))-1
    data = pd.read_csv(os.path.abspath(os.path.join(HOME_DIR, 'api_data_{}/{}_api_predicting_data.csv'.format(data_frequency, file_id))))[:no_of_datapoints]


    # print (len(data))
    data.to_csv(os.path.abspath(os.path.join(HOME_DIR, 'config{}_files/predicting/0_raw.csv'.format(config_id))))
    config = pd.read_csv(os.path.abspath(os.path.join(HOME_DIR, 'configs.csv')), index_col='config_id').loc[config_id]
    target = config['target_parameter']

    # CHANGE THIS FTLOG
    # output = pd.read_csv(os.path.abspath(os.path.join(HOME_DIR, '.output_{}.csv'.format(config_id)), index_col='timestamp')    
    output = pd.read_csv(os.path.abspath(os.path.join(HOME_DIR, 'config{}_files/output.csv'.format(config_id))), index_col='timestamp')    
    last_timestamp = int(data.iloc[-1:]['timestamp'])
    output.loc[last_timestamp, target] =  float(data.iloc[-1:][target])
    #!!!!!!!!!!!!!!!!!!
    # output.to_csv(os.path.abspath(os.path.join(HOME_DIR, '.output_{}.csv'.format(config_id)))#, mode='a', header=False)
    output.to_csv(os.path.abspath(os.path.join(HOME_DIR, 'config{}_files/output.csv'.format(config_id))))#, mode='a', header=False)
    #!!!!!!!!!!!!!!!FFS
    return None

def pad_and_sort_predicting_data(config_id, HOME_DIR):

    input_data_file = os.path.abspath(os.path.join(HOME_DIR, "config{}_files/predicting/0_raw.csv".format(config_id)))
    output_data_file = os.path.abspath(os.path.join(HOME_DIR, "config{}_files/predicting/1_padded_and_sorted_predicting_data.csv".format(config_id)))

    data = pd.read_csv(input_data_file)
    data = data.sort_values(by='timestamp', ascending=True)
    data = data.fillna(method='ffill')
    data = data.fillna(0)

    data.to_csv(output_data_file, header=True, index=False)
    return None


def label_data_targets_and_inputs(config_id, HOME_DIR):

    input_data_file = os.path.abspath(os.path.join(HOME_DIR, "config{}_files/predicting/1_padded_and_sorted_predicting_data.csv".format(config_id)))
    output_data_file = os.path.abspath(os.path.join(HOME_DIR, "config{}_files/predicting/2_labelled_predicting_data.csv".format(config_id)))
    
    data = pd.read_csv(input_data_file)
    
    config = pd.read_csv(os.path.abspath(os.path.join(HOME_DIR, 'configs.csv')), index_col='config_id').loc[config_id]
    
        
    target = config['target_parameter']
    threshold = config['target_threshold']
    horizon = config['horizon']

    prev_target_values = list(data[target])[:-horizon]
    data = data[horizon:]

    if not os.path.exists('targets.csv'):
        data['prev_{}'.format(target)] = prev_target_values
        data['delta_{}'.format(target)] = data[target].apply(lambda x: float(x)) - data['prev_{}'.format(target)].apply(lambda x: float(x))
        data['delta_percent_{}'.format(target)] = data['delta_{}'.format(target)].apply(lambda x: float(x)) / data[target].apply(lambda x: float(x))


    data.to_csv(output_data_file, header=True, index=False)
    return None

def normalize_predicting_data(config_id, HOME_DIR):

    input_data_file = os.path.abspath(os.path.join(HOME_DIR, "config{}_files/predicting/2_labelled_predicting_data.csv".format(config_id)))
    output_data_file = os.path.abspath(os.path.join(HOME_DIR, "config{}_files/predicting/3_normalized_predicting_data.csv".format(config_id)))
    reference_data_file = os.path.abspath(os.path.join(HOME_DIR, "config{}_files/training/2_labelled_training_data.csv".format(config_id)))

    data = pd.read_csv(input_data_file)
    ref_data = pd.read_csv(reference_data_file)

    ref_data = ref_data.drop(columns=['target', 'target_inverted', 'case_study'])


    for col in ref_data.columns:
        try:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            ref_data[col] = pd.to_numeric(ref_data[col], errors='coerce')

            min_value = float(ref_data[col].min()) 
            max_value = float(ref_data[col].max())
            data[col] = (data[col].apply(lambda x: float(x)) - min_value)/ (max_value-min_value)
        except:#pass
            print ("couldn't normalize {}".format(col)) 
            # data[col] = 0


    data.to_csv(output_data_file, header=True, index=False)
    return None

def sequence_predicting_data(config_id, HOME_DIR):

    input_data_file = os.path.abspath(os.path.join(HOME_DIR, "config{}_files/predicting/3_normalized_predicting_data.csv".format(config_id)))
    output_data_file = os.path.abspath(os.path.join(HOME_DIR, "config{}_files/predicting/4_sequenced_predicting_data.csv".format(config_id)))
    
    data = pd.read_csv(input_data_file)
    config = pd.read_csv(os.path.abspath(os.path.join(HOME_DIR, 'configs.csv')), index_col='config_id').loc[config_id]

    sequenced_data = pd.DataFrame()
    total_pnts = len(data)
    n_seq = 0
    seq_size = config['sequence_size']
    horizon = config['horizon']

    print (colored('sequencing...', 'magenta'))
    for i in range(0, total_pnts-seq_size-1+1):
        temp = data[i:i+seq_size+1]
        n_seq += 1
        temp['seq_id'] = n_seq
        temp['seq_pos'] = range(-seq_size, 1)
        sequenced_data = sequenced_data.append(temp, ignore_index=True)
        print (colored(i, 'magenta'), end='\r')

    sequenced_data.to_csv(output_data_file, index=False)
    return None

    # try:
    #     input_mask = sequenced_data['seq_pos'] <= 0
    #     sequenced_data = sequenced_data[input_mask] 
    # except:pass
    # # print(sequenced_data)

    # # data.to_csv(output_data_file, header=True, index=False)
    # return None


