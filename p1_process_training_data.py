
import time
import pandas as pd
import numpy as np
import os
import requests
import random
from pandas import json_normalize
from termcolor import colored

from datetime import datetime


# try:
#     from ...bot import Bot
# except:pass
# try:
#     from ..bot import Bot
# except:pass

# HOME_DIR = os.getcwd()+'/bots/models/models/algos/orthos/'

def process_training_data(config_id, bot_id, HOME_DIR):
    create_folder_tree(config_id, HOME_DIR)
    api_fetch_data(config_id, HOME_DIR)
    pad_and_sort_training_data(config_id, HOME_DIR)
    label_data_targets_and_inputs(config_id, HOME_DIR)
    normalize_training_data(config_id, HOME_DIR)
    sequence_training_data(config_id, bot_id, HOME_DIR)
    split_into_learning_testing_and_verifing_data(config_id, HOME_DIR)

    return None

def create_folder_tree(config_id, HOME_DIR):
    if os.path.exists(os.path.abspath(os.path.join(HOME_DIR, "config{}_files".format(config_id)))):
        return None
    os.mkdir(os.path.abspath(os.path.join(HOME_DIR, "config{}_files".format(config_id))))
    os.mkdir(os.path.abspath(os.path.join(HOME_DIR, "config{}_files/training".format(config_id))))
    os.mkdir(os.path.abspath(os.path.join(HOME_DIR, "config{}_files/verifying".format(config_id))))
    os.mkdir(os.path.abspath(os.path.join(HOME_DIR , "config{}_files/predicting".format(config_id))))

    cases = pd.read_csv(os.path.abspath(os.path.join(HOME_DIR,'case_studies.csv')))
    # for n in range (len(cases)):
    #     os.mkdir(HOME_DIR + "config{}_files/case_study_{}".format(config_id, n))

    return None

def api_fetch_data(config_id, HOME_DIR):

    training_output_data_file = os.path.abspath(os.path.join(HOME_DIR, "config{}_files/training/0_api_training_data.csv".format(config_id)))
    if os.path.exists(training_output_data_file):
        return None

    config = pd.read_csv(os.path.abspath(os.path.join(HOME_DIR,'configs.csv')), index_col='config_id').loc[int(config_id)]

    no_of_datapoints = config['no_of_datapoints'].astype(np.int64)
    data_frequency = config['data_frequency']

    data = pd.read_csv(os.path.abspath(os.path.join(HOME_DIR,'api_data_{}/0_api_training_data.csv'.format(data_frequency))))[:no_of_datapoints-1]

    data.to_csv(training_output_data_file)

    return None


def pad_and_sort_training_data(config_id, HOME_DIR):

    input_data_file = os.path.abspath(os.path.join(HOME_DIR, "config{}_files/training/0_api_training_data.csv".format(config_id)))
    output_data_file = os.path.abspath(os.path.join(HOME_DIR, "config{}_files/training/1_padded_and_sorted_training_data.csv".format(config_id)))
    if os.path.exists(output_data_file):
        return None

    data = pd.read_csv(input_data_file, index_col='t')

    data = data.sort_values(by='t', ascending=True)
    data = data.fillna(method='ffill')
    data = data.fillna(0)

    data.to_csv(output_data_file, index=False)
    return None

def label_data_targets_and_inputs(config_id, HOME_DIR):

    input_data_file = os.path.abspath(os.path.join(HOME_DIR, "config{}_files/training/1_padded_and_sorted_training_data.csv".format(config_id)))
    output_data_file = os.path.abspath(os.path.join(HOME_DIR, "config{}_files/training/2_labelled_training_data.csv".format(config_id)))

    if os.path.exists(output_data_file):
        return None
    data = pd.read_csv(input_data_file)

    config = pd.read_csv(os.path.abspath(os.path.join(HOME_DIR, 'configs.csv')) , index_col='config_id').loc[config_id]

    if not os.path.exists('targets.csv'):
        
        target = config['target_parameter']
        threshold = config['target_threshold']
        horizon = 1 #config['horizon'].astype(np.int64)

        prev_target_values = list(data[target])[:-horizon]
        data = data[horizon:]

        data['prev_{}'.format(target)] = prev_target_values
        data['delta_{}'.format(target)] = data[target].apply(lambda x: float(x)) - data['prev_{}'.format(target)].apply(lambda x: float(x))
        data['delta_percent_{}'.format(target)] = data['delta_{}'.format(target)].apply(lambda x: float(x)) / data[target].apply(lambda x: float(x))

        data['target'] = np.where(data['delta_percent_{}'.format(target)]>threshold, 1, 0)
        data['target_inverted'] = np.where(data['delta_percent_{}'.format(target)]>threshold, 0, 1)


    else:
        targets = pd.read_csv('targets.csv', index_col='t')
        # print (targets['t'])
        # print (data)
        data = data.merge(targets, how='left')
        print (data['target'])
        data.dropna(axis=0, subset=['target'], inplace=True)
        print (data['target'])
        data['target_inverted'] = np.where(data['target']>0, 1, 0)



    # data.to_csv(output_data_file, index=False)

    # data = pd.read_csv(input_data_file)
    data["case_study"] = np.full(len(data), -1)
    case_studies = pd.read_csv(os.path.abspath(os.path.join(HOME_DIR, "case_studies.csv")), index_col='id')
    for n in range(len(case_studies)):
        # print (case)
        case_study_timestamp_start = int(datetime.timestamp(datetime.strptime(case_studies.loc[n, 'start_date'], '%Y-%m-%d')))
        case_study_timestamp_end = int(datetime.timestamp(datetime.strptime(case_studies.loc[n, 'end_date'], '%Y-%m-%d')))
        print (case_study_timestamp_start)
        print (case_study_timestamp_end)
        # verification_mask = data['timestamp'].isin(range(case_study_timestamp_start, case_study_timestamp_end))
        # data[verification_mask] = n
        data["case_study"] = np.where(data['timestamp'].isin(range(case_study_timestamp_start, case_study_timestamp_end)), n, data["case_study"])
        #data["case_study"].apply(lambda x: if(data['timestamp'].isin(range(case_study_timestamp_start, case_study_timestamp_end))): n else: x)

        # verification_data = verification_data.append(data[verification_mask])
        # data = data[~verification_mask]

    # print (data["case_study"].max())
    # verification_data.to_csv(output_verification_data_file, index=False)
    data.to_csv(output_data_file, index=False)
    
    configs = pd.read_csv(os.path.abspath(os.path.join(HOME_DIR, 'configs.csv')))
    configs.loc[config_id, "postive_results"] = data['target'].sum()
    configs.loc[config_id, "negative_results"] = data['target_inverted'].sum()
    configs.to_csv(os.path.abspath(os.path.join(HOME_DIR,'configs.csv')), index=False)

    return None


def normalize_training_data(config_id, HOME_DIR):

    input_data_file = os.path.abspath(os.path.join(HOME_DIR, "config{}_files/training/2_labelled_training_data.csv".format(config_id)))
    output_data_file = os.path.abspath(os.path.join(HOME_DIR, "config{}_files/training/3_normalized_training_data.csv".format(config_id)))
    # output_data_verification_file = HOME_DIR + "config{}_files/training/3_normalized_training_data.csv".format(config_id)

    if os.path.exists(output_data_file):
        return None

    data = pd.read_csv(input_data_file)

    for col in data.columns:
        if col == 'case_study':
            pass
        else:
            data[col] = pd.to_numeric(data[col], errors='coerce')

            min_value = float(data[col].min()) 
            max_value = float(data[col].max())
            data[col] = (data[col].apply(lambda x: float(x)) - min_value)/ (max_value-min_value)


    # verification_data = pd.DataFrame

    data.to_csv(output_data_file, index=False)
    # print (data)
    return None


def sequence_training_data(config_id, bot_id, HOME_DIR):

    input_data_file = os.path.abspath(os.path.join(HOME_DIR, "config{}_files/training/3_normalized_training_data.csv".format(config_id)))
    output_data_file = os.path.abspath(os.path.join(HOME_DIR, "config{}_files/training/4_sequenced_training_data.csv".format(config_id)))

    if os.path.exists(output_data_file):
        return None
    data = pd.read_csv(input_data_file)

    config = pd.read_csv(os.path.abspath(os.path.join(HOME_DIR, 'configs.csv')), index_col='config_id').loc[config_id]

    sequenced_data = pd.DataFrame()
    total_pnts = len(data)
    n_seq = 0
    seq_size = config['sequence_size'].astype(np.int64)
    horizon = config['horizon'].astype(np.int64)

    print (colored('sequencing...', 'magenta'))
    # bot = Bot.objects.filter(bot_id = bot_id).last()
    # bot.set_live_feed("sequencing...", 'grey')



    first_data = data[0:seq_size+horizon]
    first_data['seq_id'] = n_seq
    first_data['seq_pos'] = range(-seq_size, horizon)
    first_data['case_study'] = first_data.iloc[-1]['case_study']

    first_data.to_csv(output_data_file, index=False)


    for i in range(1, total_pnts-seq_size-horizon+1):
        sequenced_data = data[i:i+seq_size+horizon]
        n_seq += 1
        sequenced_data['seq_id'] = n_seq
        sequenced_data['seq_pos'] = range(-seq_size, horizon)

        # if temp['case_study'].max() > -1:
        sequenced_data['case_study'] = sequenced_data.iloc[-1]['case_study']
        sequenced_data.to_csv(output_data_file, mode='a', header=False, index=False)

        # sequenced_data = sequenced_data.append(temp, ignore_index=True)
        print (colored(i, 'magenta'), end='\r')
        # bot.set_live_feed("sequencing {}".format(i), 'lightsteelblue', new_line=False)

    # sequenced_data.to_csv(output_data_file, index=False)
    return None

def split_into_learning_testing_and_verifing_data(config_id, HOME_DIR):

    input_data_file = os.path.abspath(os.path.join(HOME_DIR, "config{}_files/training/4_sequenced_training_data.csv".format(config_id)))
    output_data_folder = os.path.abspath(os.path.join(HOME_DIR, "config{}_files/training/".format(config_id)))

    if os.path.exists(os.path.abspath(os.path.join(output_data_folder,'5_processed_learning_inputs_data.csv'))):
        return None
    data = pd.read_csv(input_data_file)
    config = pd.read_csv(os.path.abspath(os.path.join(HOME_DIR, 'configs.csv')), index_col='config_id').loc[config_id]

    for col in data.columns:
        if 'Unnamed' in col:
            data = data.drop(columns=[col])
            # print (col)

        if 'partition' in col:
            data = data.drop(columns=[col])
            # print (col)


    learning_coeff, testing_coeff= config['data_split_coeffs'].split('_')
    learning_coeff = float(learning_coeff)
    testing_coeff = float(testing_coeff)
    # output_coeff = float(output_coeff)
    sum_coeff = learning_coeff + testing_coeff# + output_coeff


    learning_begin = 0  
    learning_end_testing_begin = int (max(data['seq_id']) * learning_coeff/sum_coeff) 
    testing_end_output_begin =int (max(data['seq_id']))-1 
    output_end = int (max(data['seq_id']))

    learning_mask = data['seq_id'].isin(range(learning_begin, learning_end_testing_begin-1))
    testing_mask = data['seq_id'].isin(range(learning_end_testing_begin, testing_end_output_begin-1))
    output_mask = data['seq_id'].isin(range(testing_end_output_begin, output_end))


    # non_verifying_mask = data['seq_id'].isin(verifying_seq_ids)

    non_verifying_mask = data['case_study'] < 0

    input_mask = data['seq_pos'] <= 0
    targets_mask = data['seq_pos'] == max(data['seq_pos'])

    learning_inputs_data = data[learning_mask & non_verifying_mask & input_mask].drop(columns=['target', 'target_inverted'])
    learning_targets_data = data[learning_mask & non_verifying_mask & targets_mask].filter(items=['target', 'target_inverted'])
    testing_inputs_data = data[testing_mask & non_verifying_mask & input_mask].drop(columns=['target', 'target_inverted'])
    testing_targets_data = data[testing_mask & non_verifying_mask & targets_mask].filter(items=['target', 'target_inverted'])
    output_inputs_data = data[output_mask & non_verifying_mask & input_mask].drop(columns=['target', 'target_inverted'])
    output_targets_data = data[output_mask & non_verifying_mask & targets_mask].filter(items=['target', 'target_inverted'])

    # print ("learning inputs data")
    # print (learning_inputs_data)

    learning_inputs_data.to_csv(os.path.abspath(os.path.join(output_data_folder, '5_processed_learning_inputs_data.csv')), index=False)
    learning_targets_data.to_csv(os.path.abspath(os.path.join(output_data_folder, '5_processed_learning_targets_data.csv')), index=False)
    testing_inputs_data.to_csv(os.path.abspath(os.path.join(output_data_folder, '5_processed_testing_inputs_data.csv')), index=False)
    testing_targets_data.to_csv(os.path.abspath(os.path.join(output_data_folder, '5_processed_testing_targets_data.csv')), index=False)
    output_inputs_data.to_csv(os.path.abspath(os.path.join(output_data_folder, '6_processed_output_inputs_data.csv')), index=False)


    case_studies = pd.read_csv(os.path.abspath(os.path.join(HOME_DIR, "case_studies.csv")), index_col='id')

    # print ("VALUE COUNTS")
    verifying_data = data[~non_verifying_mask]
    # print (verifying_data['case_study'].value_counts().index.tolist())

    for n in (verifying_data['case_study'].value_counts().index.tolist()):


        verifying_mask = verifying_data['case_study'] == n


        verifying_inputs_data = verifying_data[verifying_mask & input_mask].drop(columns=['target', 'target_inverted'])
        # verifying_targets_data = verifying_data[verifying_mask & targets_mask].filter(items=['target', 'target_inverted'])
        # print ("verifiying inputs data")

        # print (verifying_inputs_data)


        verifying_inputs_data.to_csv(os.path.abspath(os.path.join(HOME_DIR, "config{}_files/verifying/case_study_{}_data.csv".format(config_id, int(n)))), index=False)
        # verifying_targets_data.to_csv(output_data_folder +'6_processed_verifying_targets_data.csv', index=False)



    config = pd.read_csv(os.path.abspath(os.path.join(HOME_DIR, 'configs.csv')), index_col='config_id').loc[config_id]    
    target = config['target_parameter']
    data = pd.read_csv(os.path.abspath(os.path.join(HOME_DIR, "config{}_files/training/2_labelled_training_data.csv".format(config_id)))).filter(items=['timestamp', target, 'case_study'])

    outputfile_data = data.iloc[-len(output_inputs_data):]
    outputfile_data.to_csv(os.path.abspath(os.path.join(HOME_DIR,'config{}_files/output.csv')).format(config_id), index=False)

    # print (data)
    for n in (data['case_study'].value_counts().index.tolist()):
        if n < 0:
            pass
        else:

            case_study_mask = data['case_study'] == n
            verify_output = data[case_study_mask]
            verify_output.to_csv(os.path.abspath(os.path.join(HOME_DIR, "config{}_files/verifying/case_study_{}_output.csv".format(config_id, n))), index=False)


    return None


if __name__ == "__main__":
    for n in range (5):
        try:
            pd.read_csv(os.path.abspath(os.path.join(HOME_DIR, 'configs.csv'))).loc[n]
            process_training_data(n)
            # train_lstm(n)
        except:
            print ("couldn't process config {}".format(n))