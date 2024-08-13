
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

# HOME_DIR = os.getcwd()+'/bots/models/models/algos/orthos/'

def train_lstm(config_id, bot_id, HOME_DIR):


    weights_folder = os.path.abspath(os.path.join(HOME_DIR, "config{}_files/weights".format(config_id)))
    processed_data_folder = os.path.abspath(os.path.join(HOME_DIR, "config{}_files/training/".format(config_id)))

    if os.path.exists(weights_folder):
        return None
    config = pd.read_csv(os.path.abspath(os.path.join(HOME_DIR, 'configs.csv')), index_col='config_id').loc[config_id]

    tf.compat.v1.disable_eager_execution()

    learning_inputs_data = pd.read_csv(os.path.abspath(os.path.join(processed_data_folder,'5_processed_learning_inputs_data.csv')))
    learning_targets_data = pd.read_csv(os.path.abspath(os.path.join(processed_data_folder,'5_processed_learning_targets_data.csv')))
    testing_inputs_data = pd.read_csv(os.path.abspath(os.path.join(processed_data_folder,'5_processed_testing_inputs_data.csv')))
    testing_targets_data = pd.read_csv(os.path.abspath(os.path.join(processed_data_folder,'5_processed_testing_targets_data.csv')))
    output_inputs_data = pd.read_csv(os.path.abspath(os.path.join(processed_data_folder,'6_processed_output_inputs_data.csv')))



    learning_inputs_data = learning_inputs_data.drop(columns=['seq_id', 'seq_pos', 'case_study']).fillna(0)
    testing_inputs_data = testing_inputs_data.drop(columns=['seq_id', 'seq_pos', 'case_study']).fillna(0)
    output_inputs_data = output_inputs_data.drop(columns=['seq_id', 'seq_pos', 'case_study']).fillna(0)


    
    N_STEP = int(config['batch_size'].astype(np.int64))*(int(config['sequence_size'].astype(np.int64))+1)
    N_BATCHES = int(len(learning_inputs_data)/N_STEP)

#TF Setup---------------------------------------------------------------------------------
    tf.compat.v1.reset_default_graph()

    with tf.compat.v1.name_scope('Input'):
        with tf.compat.v1.name_scope('InputData'):
            data_list = tf.compat.v1.placeholder(tf.float32,[None,int(learning_inputs_data.shape[1])], name = "DataList")
        with tf.compat.v1.name_scope('Reshape'):
            data_struct = tf.reshape(data_list, [-1, int(config['sequence_size'].astype(np.int64))+1, int(learning_inputs_data.shape[1])])

    with tf.compat.v1.name_scope('Result'):
        target = tf.compat.v1.placeholder(tf.float32,[None,2], name = "Target")

#-----------------------------------------------------------------------------------------
    
    with tf.compat.v1.name_scope('LSTMlayer'):
        cell = tf.keras.layers.LSTMCell(config['hidden_layers'].astype(np.int64))

    with tf.compat.v1.name_scope('RNNlayer'):
        with tf.compat.v1.name_scope('DynamicRNN'):
            value, state = tf.compat.v1.nn.dynamic_rnn(cell, data_struct, dtype = tf.float32)
        with tf.compat.v1.name_scope('Transponse'):
            value_transposed = tf.transpose(a=value,perm=[1,0,2])
        with tf.compat.v1.name_scope('Gather'):
            last = tf.gather(value_transposed, int(value_transposed.get_shape()[0]-1))

#-----------------------------------------------------------------------------------------

    weights = tf.Variable(tf.random.normal([config['hidden_layers'].astype(np.int64),int(target.get_shape()[1])]), name = "W")
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

    # bot = Bot.objects.filter(bot_id = bot_id).last()

    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        saver = tf.compat.v1.train.Saver()

        best = 100
        for i in range(int(config['total_epochs'])):
            pntr = 0
            while pntr < N_BATCHES: 

                #skip forward one data set if random number is less than dropout ratio 
                if np.random.choice(10,1) < config['dropout_ratio']*10:
                    pntr += 1

                input_data = learning_inputs_data[pntr*N_STEP:(pntr+1)*N_STEP]
                target_data = learning_targets_data[pntr*config['batch_size'].astype(np.int64):(pntr+1)*config['batch_size'].astype(np.int64)]

                if len(input_data) == N_STEP:
                    _, summary = sess.run([minimize, merged_summary],{data_list:input_data, target: target_data})
                pntr += 1 
            
            incorrect = sess.run(error,{data_list:testing_inputs_data, target:testing_targets_data})
            
            print (('Epoch {:2d} accuracy {:3.10f}%'.format(i+1,100-100*incorrect)))
            # bot.set_live_feed(('Epoch {:2d} accuracy {:3.10f}%'.format(i+1,100-100*incorrect)), 'lightsteelblue')


            if incorrect < best:
                best = incorrect
                best_i = i
                save_path = saver.save(sess, os.path.abspath(os.path.join(weights_folder, 'model.ckpt')))
                print("Model saved in path: %s" % save_path)

                # bot.set_live_feed("Model saved in path: %s" % save_path, 'grey')

        configs = pd.read_csv(os.path.abspath(os.path.join(HOME_DIR, 'configs.csv')))

        configs.loc[config_id, "session_results"] = 100-100*best
        configs.loc[config_id, "session_best_epoch"] = best_i+1

        configs.to_csv(os.path.abspath(os.path.join(HOME_DIR, 'configs.csv')), index=False)


        cases = pd.read_csv(os.path.abspath(os.path.join(HOME_DIR, "case_studies.csv")))
        for n in range(len(cases)):

            verifying_data_input = pd.read_csv(os.path.abspath(os.path.join(HOME_DIR, "config{}_files/verifying/case_study_{}_data.csv".format(config_id, n))))
            verifying_data_input = verifying_data_input.drop(columns=['seq_id', 'seq_pos', 'case_study']).fillna(0)
            verifying_data_output = pd.read_csv(os.path.abspath(os.path.join(HOME_DIR, "config{}_files/verifying/case_study_{}_output.csv".format(config_id, n))))
            
            verification = sess.run(prediction, {data_list:verifying_data_input})
            verification = pd.DataFrame(verification)

            verifying_data_output['prediction_{}'.format(config_id)] = verification[0]

            verifying_data_output.to_csv(os.path.abspath(os.path.join(HOME_DIR, "config{}_files/verifying/case_study_{}_output.csv".format(config_id, n))), index=False)
            # print ('verification')
            # print (verification)
                


        # output_data = testing_inputs_data[-len(output)*(config['sequence_size']+config['horizon']-1):]

        output = pd.read_csv(os.path.abspath(os.path.join(HOME_DIR, 'config{}_files/output.csv'.format(config_id))))#

        output_results = sess.run(prediction, {data_list:output_inputs_data})
        output_results = pd.DataFrame(output_results)

        output['prediction_{}'.format(config_id)] = output_results[0]

        sess.close()

        output.to_csv(os.path.abspath(os.path.join(HOME_DIR, 'config{}_files/output.csv'.format(config_id))), index=False)
        
        # output.to_csv(os.path.abspath(os.path.join(HOME_DIR, '.output_{}.csv'.format(config_id)), index=False)
        

    return None




if __name__ == "__main__":
    train_lstm(0)