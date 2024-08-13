
import time
import pandas as pd
import numpy as np
import os
import requests
import tensorflow as tf
import random
try:
    from ...bot import Bot
except:pass
try:
    from ..bot import Bot
except:pass 


from termcolor import colored



# HOME_DIR = os.getcwd()+'/bots/models/models/algos/orthos/'

def predict_with_lstm(config_id, bot_id, HOME_DIR):

    
    weights_folder = os.path.abspath(os.path.join(HOME_DIR,  "config{}_files/weights".format(config_id)))

    input_data_file = os.path.abspath(os.path.join(HOME_DIR,  "config{}_files/predicting/4_sequenced_predicting_data.csv".format(config_id)))
    output_data_file = os.path.abspath(os.path.join(HOME_DIR,  "config{}_files/predicting/5_prediction_data.csv".format(config_id)))

    data = pd.read_csv(input_data_file).fillna(0)   
    config = pd.read_csv(os.path.abspath(os.path.join(HOME_DIR,  'configs.csv')), index_col='config_id').loc[config_id]

    try:
        # data = data.drop(columns=['Unnamed: 0'])
        data = data[data.columns.drop(list(data.filter(regex='Unnamed:')))]
    except:pass
    try:
        data = data.drop(columns=['t'])
    except:pass
    try:
        data = data.drop(columns=['seq_id'])
    except:pass
    try:
        data = data.drop(columns=['seq_pos'])
    except:pass
    # print (data.head())
    # print (input_data.shape[1])


    tf.compat.v1.disable_eager_execution()

#TF Setup---------------------------------------------------------------------------------
    tf.compat.v1.reset_default_graph()

    with tf.compat.v1.name_scope('Input'):
        with tf.compat.v1.name_scope('InputData'):
            data_list = tf.compat.v1.placeholder(tf.float32,[None,int(data.shape[1])], name = "DataList")
        with tf.compat.v1.name_scope('Reshape'):
            data_struct = tf.reshape(data_list, [-1, int(config['sequence_size'])+1, int(data.shape[1])])

    with tf.compat.v1.name_scope('Result'):
        target = tf.compat.v1.placeholder(tf.float32,[None,2], name = "Target")

#-----------------------------------------------------------------------------------------
    
    with tf.compat.v1.name_scope('LSTMlayer'):
        cell = tf.keras.layers.LSTMCell(config['hidden_layers'])

    with tf.compat.v1.name_scope('RNNlayer'):
        with tf.compat.v1.name_scope('DynamicRNN'):
            value, state = tf.compat.v1.nn.dynamic_rnn(cell, data_struct, dtype = tf.float32)
        with tf.compat.v1.name_scope('Transponse'):
            value_transposed = tf.transpose(a=value,perm=[1,0,2])
        with tf.compat.v1.name_scope('Gather'):
            last = tf.gather(value_transposed, int(value_transposed.get_shape()[0]-1))

#-----------------------------------------------------------------------------------------

    weights = tf.Variable(tf.random.normal([config['hidden_layers'],int(target.get_shape()[1])]), name = "W")
    bias = tf.Variable(tf.constant(1.0,shape = [target.get_shape()[1]]), name = "B")

#-----------------------------------------------------------------------------------------

    with tf.compat.v1.name_scope('OutputLayer'):
        prediction = tf.nn.softmax(tf.matmul(last, weights)+bias+1e-6)

    with tf.compat.v1.name_scope('CrossEntropy'):
        cross_entropy = -tf.reduce_sum(input_tensor=target*tf.math.log(tf.clip_by_value(prediction,1e-10,1.0)))
    
    with tf.compat.v1.name_scope('Optimizer'):
        with tf.compat.v1.name_scope('Adam'):
#-------------------------------CHECKITOUT------------------------------------------

            # Define the learning rate variable
            learning_rate = tf.Variable(config['learning_rate'], trainable=False)

            # During training, update the learning rate as needed
            # For example, set a new learning rate of 0.0001
            tf.keras.backend.set_value(learning_rate, 0.0001)

#-------------------------------CHECKITOUT------------------------------------------

            optimizer  = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        with tf.compat.v1.name_scope('Minimize'):
            minimize = optimizer.minimize(cross_entropy)

    with tf.compat.v1.name_scope('Accuracy'):
        with tf.compat.v1.name_scope('NotEqual'):
            mistakes = tf.not_equal(tf.argmax(input=target,axis=1), tf.argmax(input=prediction,axis=1))
        with tf.compat.v1.name_scope('ReduceMean'):
            error = tf.reduce_mean(input_tensor=tf.cast(mistakes, tf.float32))

    init_op = tf.compat.v1.global_variables_initializer()

# Tensorboard summaries-------------------------------------------------------------------
    tf.compat.v1.summary.scalar("CrossEntropy",cross_entropy)
    tf.compat.v1.summary.scalar("Error", error)

    tf.compat.v1.summary.histogram("Weights", weights)
    tf.compat.v1.summary.histogram("Biases", bias)
    tf.compat.v1.summary.histogram("Activation", prediction)
#-----------------------------------------------------------------------------------------

    merged_summary = tf.compat.v1.summary.merge_all()

    checkpoint_path = os.path.abspath(os.path.join(HOME_DIR,  "config{}_files/weights/model.ckpt".format(config_id)))

    with tf.compat.v1.Session() as sess:
        sess.run(init_op)

        # create a saver object to restore the variables
        saver = tf.compat.v1.train.Saver()
        # restore the variables from the checkpoint file
        saver.restore(sess, checkpoint_path)
        

        prediction_output = sess.run(prediction, {data_list:data})
        prediction_output = pd.DataFrame(prediction_output)

        print (colored("config {}".format(config_id), 'black', 'on_white'))
        
        print (colored("current prediction: {:.2f}".format(prediction_output.iloc[0][0]), 'magenta'))
        # bot = Bot.objects.filter(bot_id = bot_id).last()
        # bot.set_live_feed("current prediction: {:.2f}".format(prediction_output.iloc[0][0]), 'lightsteelblue')


        print (colored("current time: {}".format(int(time.time())), 'blue'))
        # bot.set_live_feed("current time: {}".format(int(time.time())), 'grey')


        output = pd.read_csv(os.path.abspath(os.path.join(HOME_DIR, 'config{}_files/output.csv'.format(config_id))) , index_col='timestamp')

        last_timestamp = output.index[-1]
        output.loc[last_timestamp, 'prediction_{}'.format(config_id)] = float(prediction_output.iloc[0][0])

        #get binance price -------------------------------------------------------------------------
        # price_data = requests.get('https://api4.binance.com/api/v3/avgPrice' ,params={'symbol':"BTCUSDT"})
        # output.loc[last_timestamp, 'binance_price'] = float(price_data.json()['price'])
        #get binance price -------------------------------------------------------------------------


        # print (output)
        # print (float(prediction_output.iloc[0][0]))
        output.to_csv(os.path.abspath(os.path.join(HOME_DIR, 'config{}_files/output.csv'.format(config_id))))#, index=False)


        sess.close()


    return None




if __name__ == "__main__":
    predict_with_lstm(0)