import pickle
from datetime import datetime
import argparse
import os
import tensorflow as tf
from tensorflow.python.client import timeline
import math
import numpy as np
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
        self.index, self.printing_step = 0, 500
        self.neg_sampled = model.neg_sampled
        self.neg_sampled_pretraining = 1 if self.neg_sampled<1 else self.neg_sampled

        self.training_data, self.test_data, self.test_list_batches = model.training_data, model.test_data, model.test_list_batches
        self.num_epochs, self.batch_size, self.vocabulary_size, self.vocabulary_size2, self.popularity_distribution = model.epoch, model.batch_size, model.vocabulary_size, model.vocabulary_size2, model.popularity_distribution
        self.seq_length, self.epoch = model.seq_length, model.epoch
        self.embedding_size, self.embedding_matrix, self.use_pretrained_embeddings = model.embedding_size, model.embedding_matrix, model.use_pretrained_embeddings
        self.adv_generator_loss, self.adv_discriminator_loss = model.adv_generator_loss, model.adv_discriminator_loss
        self.adv_negD, self.random_negD = model.negD
        self.discriminator_type = model.D_type
        self.discriminator_samples_type = model.discriminator_samples_type
        self.one_guy_sample = np.random.choice(self.vocabulary_size-1)
        self.main_scoreD, self.confintD, self.p1D, self.confintp1D, self.p1pD, self.confintp1pD, self.true_neg_prop = \
            list(), list(), list(), list(), list(), list(), list()

    def create_graph(self):
        self.train_words, self.label_words, self.dropout, self.popularity_distribution_tensor, self.context_words = \
            create_placeholders(self)
        create_discriminator(self)
        self.before_softmax_G = self.before_softmax_D
        if (self.model_params.model_type=="selfplay") or (self.model_params.model_type=="uniform") :
            self.d_loss2 = -discriminator_adversarial_loss(self)
            self.d_loss = 0.5*(self.d_loss1+self.d_loss2) if (self.adv_discriminator_loss[1]=="Mixed") else self.d_loss2
            self.disc_optimizer_adv, self.adv_grad = tf.train.AdamOptimizer(0.75e-3, beta1=0.8 ,beta2= 0.9, epsilon=1e-5), tf.gradients(self.d_loss, self.d_weights)
            self.d_train_adversarial = self.disc_optimizer_adv.minimize(self.d_loss, var_list=self.d_weights)
        
        self.disc_optimizer = tf.train.AdamOptimizer(0.75e-3, beta1=0.8 ,beta2= 0.9, epsilon=1e-5)
        self.d_baseline = self.disc_optimizer.minimize(self.d_loss1, var_list=self.d_weights)
        self.d_softmax, self.d_mle = self.disc_optimizer.minimize(self.softmax_loss, var_list=self.d_weights), self.disc_optimizer.minimize(-self.mle_lossD, var_list=self.d_weights)
        self.softmax_grad = tf.gradients(self.softmax_loss, self.d_weights)

    def train_model_with_tensorflow(self):
        self.create_graph()
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())
        step, cont = 0, True
        self.Gen_loss1, self.Gen_loss2, self.Disc_loss1, self.Disc_loss2, self.pic_number = 0, 0, 0, 0, 0
        disc_loss1, disc_loss2 = 0, 0

        #self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) #self.run_metadata = tf.RunMetadata()
        while cont:
            try:
                if (np.random.uniform()<self.neg_sampled):
                    if (self.model_params.model_type=="baseline"):
                        _, disc_loss1 = training_step(self, list_of_operations_to_run=[self.d_baseline, self.d_loss1])
                    elif (self.model_params.model_type=="softmax"):
                        _, disc_loss1 = training_step(self, list_of_operations_to_run=[self.d_softmax, self.softmax_loss])
                    elif (self.model_params.model_type=="MLE"):
                        _, disc_loss1 = training_step(self, list_of_operations_to_run=[self.d_mle, -self.mle_lossD])
                    else:
                        _, disc_loss1, disc_loss2 = training_step(self, list_of_operations_to_run=[self.d_train_adversarial, self.d_loss1, self.d_loss2])
                else:
                    _, disc_loss1 = training_step(self, list_of_operations_to_run=[self.d_mle, -self.mle_lossD])
                
                self.Disc_loss1, self.Disc_loss2 = (self.Disc_loss1+disc_loss1, self.Disc_loss2+disc_loss2)

                if (math.isnan(disc_loss1)) or (math.isnan(disc_loss2)):
                    cont = False
                if (step > self.model_params.min_steps) and early_stopping(self.main_scoreD, 15):
                    cont = False
                    
                cont, step = testing_step(self, step, cont), step+1
            
            except KeyboardInterrupt:
                self.save_data()
                raise

        self.save_data()
    
    def save_data(self):
        data = np.array([self.main_scoreD, self.confintD, self.p1D, self.confintp1D, self.p1pD, self.confintp1pD, self.true_neg_prop])
        print(data)
        if (self.model_params.metric=="MPR") or (self.model_params.metric=="AUC"):
            np.save(self.model_params.name, data)
