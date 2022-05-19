# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 16:40:08 2020
last modified: April 19, 2022
"""
# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
#load modules
import tensorflow as tf
import numpy as np
#import pandas as pd
import csv
import os
from datetime import datetime
import time
#import sys
#import pdb

os.environ["CUDA_VISIBLE_DEVICES"]="0"
tf.reset_default_graph()

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs" #directory for saving the summary information
logdir = "{}/run-{}/".format(root_logdir, now)
start_time=time.time()

# Specify the name of the checkpoints directory
checkpoint_dir = "save_noGT_1samples"

# Create the directory if it does not already exist
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Specify the path to the checkpoint file
model_path = checkpoint_dir+"/mymodel"

class Model(object):
    
    def __init__(self,input_dim=25, T=180, D=501, prev=16,
                 lstm_size=128,
                 batch_size=100, e_learning_rate=1e-4, dropout_rate=0.5,
                 ):
        self.input_dim = input_dim
        self.T = T
        self.D = D
        self.prev = prev

        self.enc_size = lstm_size 
        
        self.batch_size = batch_size
        self.e_learning_rate = e_learning_rate
        self.dropout_rate=dropout_rate

        self._srng = np.random.RandomState(np.random.randint(1,2147462579))
        self.he_init = tf.contrib.layers.variance_scaling_initializer()
        self.lstm_enc = tf.contrib.rnn.LSTMCell(self.enc_size, state_is_tuple=True)
        
        
        # build computation graph of model
        self.DO_SHARE=None
        self.x = tf.placeholder(tf.float32, shape=[self.batch_size, self.input_dim],name="input_x")
        #self.ymax = tf.placeholder(tf.float32, shape=[self.batch_size, 1],name="ymax")
        self.y = tf.placeholder(tf.int32, shape=[self.batch_size, self.T],name="y")
        self.training = tf.placeholder_with_default(False, shape=[], name='training')
        
        # initial state
        self.enc_state = self.lstm_enc.zero_state(self.batch_size, tf.float32)
        self.yss = [0.5] * self.D
        self.num_count  = [0] * self.D
        self.y_prev = 0.0
        #self.e_loss = 0.0
        
        xe = self.input_embedding(self.x)
        
        for t in range(self.D): #thedistribution is predicted in a reverse way, so t=0 is for mRNA=500, t=500 is for mRNA=0
            
            self.y_prev = self.get_yprev(t)
            h_enc, self.enc_state = self.encode(self.enc_state, tf.concat([xe, self.y_prev], 1))
            self.yss[t] = self.linear(h_enc)
            #self.yss[t] = tf.sigmoid(ylt)
            self.num_count[t] = tf.reduce_sum(tf.cast(tf.equal(self.y, D-1-t), tf.float32), 1)
            #pdb.set_trace()
            self.DO_SHARE = True
        self.y_logits=tf.squeeze(tf.stack(self.yss,axis=1))
        self.y_pred=tf.nn.softmax(self.y_logits)
        ncount=tf.stack(self.num_count, axis=1)
        #pdb.set_trace()
        self.e_loss=tf.reduce_sum(-tf.multiply(ncount, tf.log(self.y_pred)))
        
        self.e_vars = tf.trainable_variables()

        self.e_optimizer = tf.train.AdamOptimizer(self.e_learning_rate, beta1=0.5, beta2=0.999)
        e_grads = self.e_optimizer.compute_gradients(self.e_loss, self.e_vars)
        #pdb.set_trace()
        clip_e_grads = [(tf.clip_by_norm(grad, 10), var) for grad, var in e_grads if grad is not None]
        self.e_optimizer = self.e_optimizer.apply_gradients(clip_e_grads)


        self.eloss_summary = tf.summary.scalar('eloss', self.e_loss)
        self.file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    
    def train(self, train_set, valid_set, maxEpoch=10):
        # Create a Saver Object
         with tf.Session() as sess:
            
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            
            i = 0
            loss_v=np.zeros((maxEpoch,4))
            for epoch in range(maxEpoch): # range for python3
                Les, Levs = [], []              
                for xtrain, ytrain in self.data_loader(train_set, self.batch_size, shuffle=True):
                    #ytrain = ytrain[:,::-1]
                    
                    _, Le, ys= sess.run([self.e_optimizer, self.e_loss, self.y_pred], 
                                     feed_dict={self.x: xtrain, self.y: ytrain})
                    Les.append(Le)
                    i += 1
                    
                    if i % 100 == 0:
                        summary_str_e = self.eloss_summary.eval(feed_dict={self.x: xtrain, self.y: ytrain})
                        self.file_writer.add_summary(summary_str_e, i)
                for xvalid, yvalid in self.data_loader(valid_set, self.batch_size):
                    #yvalid = yvalid[:,::-1]
                    
                    Lev,ysv = sess.run([self.e_loss, self.y_pred], feed_dict={self.x: xvalid, self.y: yvalid})
                    Levs.append(Lev)
                Le_train_mean = np.array(Les).mean()
                Le_valid_mean = np.array(Levs).mean()
                Le_train_std = np.array(Les).std()
                Le_valid_std = np.array(Levs).std()

                loss_v[epoch,:]=[Le_train_mean, Le_valid_mean,  Le_train_std, Le_valid_std]   
                print("Epoch:",epoch,"\t train loss:",Le_train_mean, "\t valid loss:",Le_valid_mean )
                saver.save(sess, model_path, global_step=epoch)
                #pdb.set_trace()                
                np.savetxt(checkpoint_dir+'/ys.txt',ys )
                np.savetxt(checkpoint_dir+'/ytrain.txt', ytrain )
                np.savetxt(checkpoint_dir+'/ysv.txt',ysv )
                np.savetxt(checkpoint_dir+'/yvalid.txt', yvalid )  
                np.savetxt(checkpoint_dir+'/xtrain.txt', xtrain )
                np.savetxt(checkpoint_dir+'/xvalid.txt', xvalid ) 
            np.savetxt(checkpoint_dir+'/loss.txt', loss_v )
            self.file_writer.close()
    


    def data_loader(self, data_set, batchsize, shuffle=False): 
        features, labels = data_set
        if shuffle:
            indices = np.arange(len(features))
            self._srng.shuffle(indices)
        for start_idx in range(0, len(features) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield features[excerpt], labels[excerpt]
    
    def lrelu(self, x, alpha=0.2):
        return tf.maximum(x, alpha*x)
    
    def encode(self, state, input):
        """
        run LSTM
        state = previous encoder state
        input = cat(read,h_dec_prev)
        returns: (output, new_state)
        """
        with tf.variable_scope("e_lstm",reuse=self.DO_SHARE):
            return self.lstm_enc(input,state)
            
    #fully_connected creates a variable called weights,
    #representing a fully connected weight matrix, which is multiplied by the inputs to produce a Tensor of hidden units
    def linear(self, x):
        with tf.variable_scope("e_linear", reuse=self.DO_SHARE):
            yl = tf.layers.dense(x, 1, activation=None)
        return yl # output logits w.r.t sigmoid
    
    def input_embedding(self, x):
#        with tf.variable_scope("e_eblinear1", reuse=None):
#            h0 = tf.layers.dense(x,512, kernel_initializer=self.he_init)
#            h1 = tf.layers.dense(h0, 256, kernel_initializer=self.he_init, activation=self.lrelu)
#            h1_drop=tf.layers.dropout(h1, self.dropout_rate, training=self.training)
        with tf.variable_scope("e_eblinear2", reuse=None):
            h2 = tf.layers.dense(x, 256, kernel_initializer=self.he_init, activation=tf.nn.relu)
        return h2
    
#    def height_model(self, x):
#        with tf.variable_scope("e_hlinear1", reuse=None):
#            h1 = tcl.fully_connected(inputs=x, num_outputs=64, activation_fn=tf.nn.relu)
#        with tf.variable_scope("e_hlinear2", reuse=None):
#            h2 = tcl.fully_connected(inputs=h1, num_outputs=1, activation_fn=None)
#        return h2

    def get_yprev(self, t):
        with tf.variable_scope("e_yprev", reuse=self.DO_SHARE):
            yp_init = tf.get_variable('yp_init', [self.batch_size, self.prev], initializer=tf.constant_initializer(0.5))
        return yp_init if t == 0 else tf.concat([self.y_prev[:,1:], self.yss[t-1]], 1)
            

if __name__ == "__main__":
    
    # TODO: preprocessing dataset
    # Load data from csv file
    with open('datas/all_data_for_training.csv') as csvfile:
        mpg = list(csv.reader(csvfile))
        results=np.array(mpg).astype("float")

    #assign 1500 data set to train set and the rest to valid set
        #data structure: 0:2 input parameters; 3:503 pdf;
    bsize=100
    num_samples=1
    train_size=int(len(results)*0.9/bsize)*bsize
    valid_size=len(results)-train_size
    
    print("train size:",train_size,"\t valid size:",valid_size)
    
    train_set = results[:train_size,0:3], np.rint(results[:train_size,3:(num_samples+3)]) #parameters,distribution,peak value
    valid_set = results[-valid_size:,0:3], np.rint(results[-valid_size:,3:(num_samples+3)])
    
    del mpg,results

    mymodel = Model(input_dim=3, T=num_samples, D=301, batch_size=bsize)
    mymodel.train(train_set, valid_set, maxEpoch=300) # # of iters = maxepoch * N/bs
