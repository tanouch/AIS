import pickle
from datetime import datetime
import argparse
import os
import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.contrib.opt import LazyAdamOptimizer

import math
import numpy as np
from scipy import stats
import time

from tools import *
from metric_utils import *
from graphs_utils import *
from training_utils import *

from gen_cnn import Gen_CNN_Model
from lstm import LSTM_Model
from word2vec import W2V_Model
from word2vec_google import *

class Self_Basket_Completion_Model(object):

    def __init__(self, model):
        self.model_params = model
        self.train_data, self.test_data, self.X_train, self.Y_train, self.X_test, self.Y_test = list(), list(), list(), list(), list(), list()
        self.LSTM_labels_train, self.LSTM_labels_test = list(), list()
        self.index, self.index_words = 0, 0
        self.neg_sampled = model.neg_sampled
        self.neg_sampled_pretraining = 1 if self.neg_sampled<1 else self.neg_sampled

        self.training_data, self.test_data = model.training_data, model.test_data
        self.num_epochs, self.batch_size, self.vocabulary_size, self.vocabulary_size2 = model.epoch, model.batch_size, model.vocabulary_size, model.vocabulary_size2
        self.seq_length, self.epoch = model.seq_length, model.epoch
        self.embedding_size, self.embedding_matrix, self.use_pretrained_embeddings = model.embedding_size, model.embedding_matrix, model.use_pretrained_embeddings
        self.adv_generator_loss, self.adv_discriminator_loss = model.adv_generator_loss, model.adv_discriminator_loss
        self.negD = model.negD
        self.discriminator_type = model.D_type
        self.one_guy_sample = np.random.choice(self.vocabulary_size-1)
        self.dataD = [list(), list(), list(), list(), list(), list(), list()]
        self.Gen_loss1, self.Gen_loss2, self.Disc_loss1, self.Disc_loss2, self.pic_number = 0, 0, 0, 0, 0

    def create_graph(self):
        create_placeholders(self)
        create_discriminator(self, size=1)
        self.before_softmax_G, self.before_softmax_embedding_G = self.before_softmax_D, self.before_softmax_embedding_D
        if (self.model_params.model_type=="SS") or (self.model_params.model_type=="BCE"):
            self.d_loss2 = -discriminator_adversarial_loss(self) #sampled_softmax_loss_improved(self)
            self.disc_optimizer_adv, self.adv_grad = LazyAdamOptimizer(1.5e-3, beta1=0.8 ,beta2= 0.9, epsilon=1e-5), tf.gradients(self.d_loss2, self.d_weights)
            self.d_train_adversarial = self.disc_optimizer_adv.minimize(self.d_loss2, var_list=self.d_weights)
        
        lr = 1e-3
        global_step = tf.Variable(0, trainable=False)
        rate = tf.train.exponential_decay(lr, global_step, 3, 0.9999)
        self.disc_optimizer = LazyAdamOptimizer(lr, beta1=0.8 ,beta2= 0.9, epsilon=1e-5)
        self.d_baseline = self.disc_optimizer.minimize(self.d_loss1, var_list=self.d_weights, global_step=global_step)
        self.d_softmax, self.d_mle = self.disc_optimizer.minimize(self.softmax_loss, var_list=self.d_weights, global_step=global_step), self.disc_optimizer.minimize(-self.mle_lossD, var_list=self.d_weights, global_step=global_step)
        self.softmax_grad = tf.gradients(self.softmax_loss, self.d_weights)

    def train_model_with_tensorflow(self):
        self.create_graph()
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())
        self.options, self.run_metadata = create_options_and_metadata(self)
        step, cont = 0, True
        disc_loss1, disc_loss2 = 0, 0
        
        timee = time.time()
        while cont:
            try:
                if (self.model_params.model_type=="baseline"):
                    _, disc_loss1 = training_step(self, [self.d_baseline, self.d_loss1])
                elif (self.model_params.model_type=="softmax"):
                    _, disc_loss1 = training_step(self, [self.d_softmax, self.softmax_loss])
                elif (self.model_params.model_type=="MLE"):
                    _, disc_loss1 = training_step(self, [self.d_mle, -self.mle_lossD])
                else:
                    _, disc_loss1, disc_loss2 = training_step(self, [self.d_train_adversarial, self.d_loss1, self.d_loss2])
                
                self.Disc_loss1, self.Disc_loss2 = (self.Disc_loss1+disc_loss1, self.Disc_loss2+disc_loss2)

                if (math.isnan(disc_loss1)) or (math.isnan(disc_loss2)):
                    cont = False

                if ((step > self.model_params.min_steps) and (early_stopping(self.dataD[0], 4) or early_stopping(self.dataD[2], 4))) or \
                    (step>self.model_params.max_steps):
                    cont = False
                
                if (step % self.model_params.printing_step==0):
                    print(time.time()-timee)

                self.save_data(step)
                testing_step(self, step)
                create_timeline_object(self)

                if (step % self.model_params.printing_step==0):
                    timee = time.time()
                step += 1
            
            except KeyboardInterrupt:
                cont = False
        
        self.save_data(step)
        tf.reset_default_graph()
    
    def save_data(self, step):
        if (step%self.model_params.saving_step==0):
            data = np.array(self.dataD)
            np.save(self.model_params.name, data)
            #np.save(self.model_params.name+"_disc_emb"+str(step), self._sess.run(self.discriminator.embeddings_tensorflow))
            #np.save(self.model_params.name+"_disc_weights"+str(step), self._sess.run(self.discriminator.nce_weights))
            #np.save(self.model_params.name+"_disc_biases"+str(step), self._sess.run(self.discriminator.nce_biases))
