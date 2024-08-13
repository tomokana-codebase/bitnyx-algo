from p1_process_training_data import process_training_data
from p2_train_lstm import train_lstm
from p3_process_predicting_data import process_predicting_data
from p4_predict_with_lstm import predict_with_lstm
from p5_process_output_data import process_case_studies_files
from p5_process_output_data import process_output_files
from p5_process_output_data import process_output_entry
from p_api_glassnode import get_api_training_data
from p_api_glassnode import get_api_predicting_data
import pandas as pd
from time import sleep, time
import os
import shutil

from termcolor import colored

try:
    from ...bot import Bot
except:pass
try:
    from ..bot import Bot
except:pass

# HOME_DIR = os.getcwd()+'/bots/models/models/algos/orthos/'


import warnings
warnings.filterwarnings("ignore")

def run_period(bot_id, HOME_DIR):
    start_time = int(time())
    
    # HOME_DIR = os.getcwd()+'/bots/models/models/bot_{}_algo/'.format(bot_id)
    configs = pd.read_csv(os.path.abspath(os.path.join(HOME_DIR, 'configs.csv')))

    # try:
    print ('running period')

    # bot.set_live_feed(line='running period', color='grey')
    print ('getting api data')
    # bot = Bot.objects.filter(bot_id = bot_id).last()
    # bot.set_live_feed("getting api data...", 'grey')
    # bot.save()

    # bot.set_live_feed(line='getting api data')
    get_api_predicting_data(configs, HOME_DIR)

    print ('got api data')
    # bot.set_live_feed(line='got api data')
    for n in range (len(configs)):

        print ('processing predicting api data')
        # bot.set_live_feed(line='processing predciting api data',color='purple')  
        process_predicting_data(n, bot_id, HOME_DIR)
        print ('predicting with lstm')
        # bot.set_live_feed(line='predciting with lstm',color='purple')
        predict_with_lstm(n,bot_id, HOME_DIR)

    data = pd.read_csv(os.path.abspath(os.path.join(HOME_DIR, 'config{}_files/output.csv'.format(n))))[-2:]
    # print (colored('price: {:.2f}'.format(data.iloc[-1]['BTC_price_usd_close']), 'light_green', 'on_blue'))

    if data.iloc[-1]['BTC_price_usd_close'] < data.iloc[-2]['BTC_price_usd_close']:
        color = "red"
    elif data.iloc[-1]['BTC_price_usd_close'] >= data.iloc[-2]['BTC_price_usd_close']:
        color = "light_green"
    print (colored('price: {:.2f}'.format(data.iloc[-1]['BTC_price_usd_close']), color, 'on_blue'))

    # bot = Bot.objects.filter(bot_id = bot_id).last()
    # bot.set_live_feed('price: {:.2f}'.format(data.iloc[-1]['BTC_price_usd_close']), color.replace('_', ''))
    # bot.save()
    # bot.set_live_feed(line='price: {:.2f}'.format(data.iloc[-1]['BTC_price_usd_close']), color=color)
        
    process_output_entry(HOME_DIR)
        # print(colored('---waiting for next cycle: {} mins---'.format(int(60*60-len(configs))), 'blue'))
        # sleep(60*60-4)
    
    # except:
    #     print ('something went wrong... trying again in 5 min')
    #     sleep(60*5)

    # start delay
    end_time = int(time())
    print (colored('**   nap time   **', 'light_blue', "on_blue"))
    
    wait_time = 60*60-(end_time-start_time)
    # start_wait_time = time()
    while ((end_time + wait_time - time()) > 0):
        print (colored(int(end_time + wait_time - time()), 'light_blue'), end='\r')
        sleep(1)

def run_training(bot_id, HOME_DIR):

    # HOME_DIR = os.getcwd()+'/bots/models/models/bot_{}_algo/'.format(bot_id)
    if not os.path.exists(HOME_DIR):
        os.mkdir(HOME_DIR)
        src_folder = os.getcwd()
        dst_folder = HOME_DIR 

        for file in os.listdir(src_folder):
            filename = os.fsdecode(file)
            if filename.endswith(".csv") or filename.endswith(".env"): 
                print(os.path.abspath(os.path.join(src_folder, filename)))
                shutil.copy(os.path.abspath(os.path.join(src_folder, filename)), dst_folder)
            # else:
            #     continue
        # shutil.copytree(src_folder, dst_folder)
    # bot = Bot.objects.filter(bot_id = bot_id).last()
    # bot.set_live_feed("starting training", 'grey')
    # bot.save()

    configs = pd.read_csv(os.path.abspath(os.path.join(HOME_DIR, 'configs.csv')))
    # trying = True

    # while trying:
        # try:
    get_api_training_data(configs, HOME_DIR)
        #     trying = False
        # except:
        #     trying = True
        #     print ('something went wrong... trying again in 5 min')
        #     sleep(60*5)

    # bot_algo_folder = os.getcwd()+'/bots/models/models/bot_{}_algo'.format(bot_id)



    for n in range (len(configs)):
        # bot = Bot.objects.filter(bot_id = bot_id).last()
        # bot.set_live_feed("processing trainging data", 'grey')
        # bot.save()

        process_training_data(n, bot_id, HOME_DIR)
        train_lstm(n, bot_id, HOME_DIR)
        # process_predicting_data(n)
        # predict_with_lstm(n)


    # output = pd.read_csv(HOME_DIR + 'output.csv')
    # for n in range (len(configs)):
    #     output_with_pred = pd.read_csv(HOME_DIR + '.output_{}.csv'.format(n))
    #     output['prediction_{}'.format(n)] = output_with_pred['prediction_{}'.format(n)]
    #     # os.remove('output_{}.csv'.format(n))
    # output.to_csv(HOME_DIR + 'output.csv', index=False)

    process_case_studies_files(HOME_DIR)
    process_output_files(HOME_DIR)
    # color = "light_green"


def run_cycle(bot_id, HOME_DIR):

    # HOME_DIR = os.getcwd()+'/bots/models/models/bot_{}_algo/'.format(bot_id)
    configs = pd.read_csv(os.path.abspath(os.path.join(HOME_DIR, 'configs.csv')))

    # state=pd.read_csv('state.csv')
    # state.loc[0,'time_of_training'] = int(time())
    # state.to_csv('state.csv')
    run_training(bot_id, HOME_DIR)
    while(True):
        run_period(bot_id, HOME_DIR)        


if __name__ == "__main__":
    run_cycle(0, os.getcwd())