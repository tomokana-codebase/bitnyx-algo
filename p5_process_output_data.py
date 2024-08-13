import time
import pandas as pd
import numpy as np
import os
import random
import math


from termcolor import colored



# HOME_DIR = os.getcwd()+'/bots/models/models/algos/orthos/'

def process_output_files(HOME_DIR):
    configs = pd.read_csv(os.path.abspath(os.path.join(HOME_DIR, 'configs.csv')), index_col='config_id')
    for k in range(len(configs)):
        file_name = os.path.abspath(os.path.join(HOME_DIR, "config{}_files/output.csv".format(k)))
        print ('processing {}'.format(file_name))
        process_file(file_name, k, HOME_DIR)


def process_case_studies_files(HOME_DIR):
    case_studies = pd.read_csv(os.path.abspath(os.path.join(HOME_DIR, "case_studies.csv")), index_col='id')
    configs = pd.read_csv(os.path.abspath(os.path.join(HOME_DIR, 'configs.csv')), index_col='config_id')
    for k in range(len(configs)):
        config = configs.iloc[k]
        for m in range(len(case_studies)):
            file_name = os.path.abspath(os.path.join(HOME_DIR, "config{}_files/verifying/case_study_{}_output.csv".format(k, m)))
            process_file(file_name, k, HOME_DIR)


def process_file(file_name, config_id, HOME_DIR):
    thres = 0.5
    configs = pd.read_csv(os.path.abspath(os.path.join(HOME_DIR, 'configs.csv')), index_col='config_id')
    config = configs.iloc[config_id]
    data = pd.read_csv(file_name)
    total_data_points = len(data)
    data['market_prediction'] = data['BTC_price_usd_close'][data['prediction_{}'.format(config_id)]>thres]

    predictions = list(data['market_prediction'])
    total_profit = 0
    total_trades = 0
    margins = [0]
    cumsum = [0]
    buys=[0]
    sells=[0]
    buy=0
    sell=0


    horizon = int(config['horizon']-1) #hardcoded to glassnode
    for n in range(len(predictions)-1):
        profit = 0
        if math.isnan(predictions[n]) and not math.isnan(predictions[n+1]):
            buy = data['BTC_price_usd_close'][n+1]
            sell = -1

        elif not math.isnan(predictions[n]) and math.isnan(predictions[n+1]) and not (buy==0):
            sell = data['BTC_price_usd_close'][n+1]
            profit  = (sell - buy)/buy*100
            if profit > 0:
                colr = "green"
            else:
                colr = "red"
            # print (colored("buy:{} sell:{}".format(buy, sell), colr))
            sells = [sell if x==-1 else x for x in sells]
            total_profit += profit
            total_trades += 1


        buys.append(buy)
        sells.append(sell)

        if profit:
            buy = 0
            sell= 0

        cumsum.append(total_profit)
        margins.append(profit)

    data['cumsum'] = cumsum
    data['margins'] = margins
    data['buys'] = buys
    data['sells'] = sells

    data.to_csv(file_name, index=False)
    # case_study_market_performance = (data['BTC_price_usd_close'][len(predictions)-1] - data['BTC_price_usd_close'][0])/data['BTC_price_usd_close'][0]*100

    # print (colored("case study {}  ----  margin:{}% . market:{}% . trades:{} . data points:{}".format(m, int(total_profit), int(case_study_market_performance), total_trades, total_data_points), 'white', 'on_black'))



def process_output_entry(HOME_DIR):

    thres = 0.5
    configs = pd.read_csv(os.path.abspath(os.path.join(HOME_DIR, 'configs.csv')), index_col='config_id')
    for k in range(len(configs)):
        config = configs.iloc[k]
        horizon = int(config['horizon'])
        data = pd.read_csv(os.path.abspath(os.path.join(HOME_DIR, 'config{}_files/output.csv'.format(k))))[-2:]

        data['market_prediction'] = data['BTC_price_usd_close'][data['prediction_{}'.format(k)]>thres]
        # print (colored('price: {:.2f}'.format(data.iloc[-1]['BTC_price_usd_close']), 'light_red', 'on_light_green'))
        predictions = list(data['market_prediction'])
        
        if math.isnan(predictions[0]) and not math.isnan(predictions[1]):
            position = 'buy @ {}'.format(int(data.iloc[-1]['BTC_price_usd_close']))
            color = 'green'
            # state=pd.read_csv('state.csv')
            config['last_entry_price'] = data.iloc[-1]['BTC_price_usd_close']
            # state.to_csv('state.csv', index=False)
        elif not math.isnan(predictions[0]) and math.isnan(predictions[1]):
            # print ('here 0')
            position = 'sell @ {}'.format(int(data.iloc[-1]['BTC_price_usd_close']))
            color = 'red'
            # print ('here 1')
            # state=pd.read_csv('state.csv')
            # buy = float(config['last_entry_price'])
            # sell = float(data.iloc[-1]['BTC_price_usd_close'])
            # config['last_entry_price'] = 0
            # print ('here 2')
            # margin = (sell - buy)/buy*100
            # print (colored('margin: {}%'.format(margin), 'cyan', 'on_black'))

            # config['total_margin'] = (margin + float(config['total_margin'])*float(config['total_trades']))/float(config['total_trades']) 
            # config['total_trades'] = float(config['total_trades']) + 1
            # # state.to_csv('state.csv', index=False)

        elif math.isnan(predictions[0]) and math.isnan(predictions[1]):
            position = 'maintain-sell'
            color = 'red'

         
        elif not math.isnan(predictions[0]) and not math.isnan(predictions[1]):
            position = 'maintain-buy'
            color = 'green'
        
        print (colored('config {} position: {}'.format(k, position), color, 'on_black'))

    process_output_files(HOME_DIR)        
    # configs.to_csv(os.path.abspath(os.path.join(HOME_DIR, 'configs.csv')

if __name__ == "__main__":
    process_case_studies_files(0)