
import time
import pandas as pd
import numpy as np
import os
import requests
import random
from pandas import json_normalize
# from train_lstm import train_lstm
from termcolor import colored

# HOME_DIR = os.getcwd()+'/bots/models/models/algos/orthos/'




def get_api_training_data(configs, HOME_DIR):

    data_frequencies = (configs['data_frequency'].unique())



    for frequency in data_frequencies:
        
        if os.path.exists(os.path.abspath(os.path.join(HOME_DIR, 'api_data_{}/0_api_training_data.csv'.format(frequency)))):
            return None

        if not os.path.exists(os.path.abspath(os.path.join(HOME_DIR, "api_data_{}".format(frequency)))):
            os.mkdir(os.path.abspath(os.path.join(HOME_DIR, "api_data_{}".format(frequency))))


        frequency_mask = configs['data_frequency'] == frequency
        config_set =  configs[frequency_mask]
        no_of_datapoints = max(config_set['no_of_datapoints'])
        
        try:
            data = get_glassnode_data(frequency, no_of_datapoints, HOME_DIR)
        except:
            print ('waiting a minute')

            time.sleep(60)
            get_api_training_data(configs, HOME_DIR)

        data.to_csv(os.path.abspath(os.path.join(HOME_DIR, 'api_data_{}/0_api_training_data.csv'.format(frequency))))
        
    return None

def get_api_predicting_data(configs, HOME_DIR):


    data_frequencies = (configs['data_frequency'].unique())

    print (colored('here','red'))
    for frequency in data_frequencies:
        print (frequency)
        file_id = len(os.listdir(os.path.abspath(os.path.join(HOME_DIR, 'api_data_{}/'.format(frequency)))))
        frequency_mask = configs['data_frequency'] == frequency
        config_set =  configs[frequency_mask]
        no_of_datapoints = max(config_set['sequence_size']) 
        

        try:
            data = get_glassnode_data(frequency, no_of_datapoints, HOME_DIR)
        except:
            print ('waiting a minute')
            time.sleep(60)
            get_api_predicting_data(configs, HOME_DIR)
        print (data)

        data.to_csv(os.path.abspath(os.path.join(HOME_DIR, 'api_data_{}/{}_api_predicting_data.csv'.format(frequency, file_id))))
        data = pd.read_csv(os.path.abspath(os.path.join(HOME_DIR, 'api_data_{}/{}_api_predicting_data.csv'.format(frequency, file_id))), index_col=False)
        for k in range(len(config_set)):

            output = data.filter(items=['timestamp','BTC_price_usd_close'])
            output.to_csv(os.path.abspath(os.path.join(HOME_DIR, 'config{}_files/output.csv'.format(k))), mode='a', header=False, index=False)

    return None






def get_glassnode_data(data_frequency, no_of_datapoints, HOME_DIR):

    with open(os.path.abspath(os.path.join(HOME_DIR, '.glassnode.env')), 'r') as f:
        API_KEY = f.read()
    asset = 'BTC'
    endpoints = pd.read_csv(os.path.abspath(os.path.join(HOME_DIR, 'params.csv')))

    if data_frequency == 'hourly':
        time_interval = '1h'
        since = int(time.time() - no_of_datapoints*60*60)
    elif data_frequency == 'daily':
        time_interval = '24h'
        since = int(time.time() - no_of_datapoints*60*60*24)
    
    until = int(time.time())

    data = pd.DataFrame()
    res = requests.get('https://api.glassnode.com'+'/v1/metrics/market/price_usd_close',
        params={'a': asset, 'i': time_interval, 's': since, 'api_key': API_KEY})

    # print (res.text)
    data = pd.read_json(res.text)
    data.set_index('t', inplace=True)
    
    print (colored('requesting data ...', 'cyan'))
    while(True):
        try:
            for n, path in enumerate(endpoints['path']):#.iloc[:100]):
                time.sleep(1)
                res = requests.get('https://api.glassnode.com'+path,
                    params={'a': asset, 'i': time_interval,'s': since, 'api_key': API_KEY})
                # print (path)
                temp = pd.read_json(res.text)
                if list(temp.columns.values)[0] != 'o':
                    temp.set_index('t', inplace=True)
                    data = pd.concat([data, temp], sort=False, axis=1, join="outer")

                    data = data.rename(columns={'v': asset+"_"+path.split('/')[-1]})
                    time.sleep(1)
                print (colored(n, 'cyan'), end='\r')    
            break
        except:pass

    data['timestamp'] = data.index
    try:
        data = data.drop(columns=['partitions'])
    except:pass


    return data
