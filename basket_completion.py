import pickle
from datetime import datetime
import argparse
import os
import tensorflow as tf
import math
import numpy as np
import time

from tools import *
from metric_utils import *
from graphs_utils import *
from training_utils import *

from lstm import LSTM_Model
from word2vec import W2V_Model
from word2vec_google import *
from gen_cnn import Gen_CNN_Model
tf.set_random_seed(42)

class Basket_Completion_Model(object):

    def __init__(self, model):
        self.model_params = model
        self.train_data, self.test_data, self.X_train, self.Y_train, self.X_test, self.Y_test = list(), list(), list(), list(), list(), list()
        self.LSTM_labels_train, self.LSTM_labels_test = list(), list()
        self.index, self.printing_step = 0, 1000

        self.neg_sampled = model.neg_sampled
        self.neg_sampled_pretraining = 1 if self.neg_sampled<1 else self.neg_sampled
        self.training_data, self.test_data = model.training_data, model.test_data
        self.num_epochs, self.batch_size, self.vocabulary_size, self.vocabulary_size2 = model.epoch, model.batch_size, model.vocabulary_size, model.vocabulary_size2
        self.seq_length, self.epoch = model.seq_length, model.epoch
        self.embedding_size, self.embedding_matrix, self.use_pretrained_embeddings = model.embedding_size, model.embedding_matrix, model.use_pretrained_embeddings
        self.adv_generator_loss, self.adv_discriminator_loss = model.adv_generator_loss, model.adv_discriminator_loss
        self.adv_negD, self.random_negD = model.negD
        self.adv_negG, self.random_negG = model.negG
        self.one_guy_sample = np.random.randint(self.vocabulary_size)

    def create_graph(self):
        create_placeholders(self)
        create_generator(self, size=1)
        create_discriminator(self, size=2)
        self.g_loss2 = -generator_adversarial_loss(self)
        self.d_loss2 = -discriminator_adversarial_loss(self)

        self.d_loss = 0.5*(self.d_loss1+self.d_loss2) if (self.adv_discriminator_loss[1]=="Mixed") else self.d_loss2
        self.g_loss = 0.5*self.g_loss2 + 0.5*self.g_loss1 if (self.adv_generator_loss[1]=="Mixed") else self.g_loss2

        lr, global_step = 1e-3, tf.Variable(0, trainable=False)
        #rate = tf.train.exponential_decay(lr, global_step, 3, 0.9999)
        self.gen_optimizer, self.gen_optimizer_adv = tf.train.AdamOptimizer(lr, beta1=0.8 ,beta2= 0.9, epsilon=1e-5), tf.train.AdamOptimizer(lr, beta1=0.8 ,beta2= 0.9, epsilon=1e-5)
        self.disc_optimizer, self.disc_optimizer_adv = tf.train.AdamOptimizer(lr, beta1=0.8 ,beta2= 0.9, epsilon=1e-5), tf.train.AdamOptimizer(lr, beta1=0.8 ,beta2= 0.9, epsilon=1e-5)
        self.d_train_adversarial = self.disc_optimizer_adv.minimize(self.d_loss, var_list=self.d_weights)
        self.g_train_MLE, self.g_train_adversarial = self.gen_optimizer.minimize(-self.mle_lossG, var_list=self.g_weights), self.gen_optimizer_adv.minimize(self.g_loss, var_list=self.g_weights, global_step=global_step)

        self.dataD = [list(), list(), list(), list(), list(), list(), list()]
        self.dataG = [list(), list(), list(), list(), list(), list(), list()]

    def train_model_with_tensorflow(self):
        self.create_graph()
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())
        step, cont = 0, True
        disc_itr = 3 if (self.model_params.dataset in ["blobs", "blobs0", "blobs1", "blobs2", "s_curve", "swiss_roll", "moons", "circles"]) else 10
        disc_loss1, disc_loss2, gen_loss1, gen_loss2, self.Gen_loss1, self.Gen_loss2, self.Disc_loss1, self.Disc_loss2, self.pic_number = 0, 0, 0, 0, 0, 0, 0, 0, 0
        disc_batch = self.batch_size

        while cont:
            try:
                if (np.random.uniform()<self.neg_sampled):
                    _, gen_loss1, gen_loss2 = training_step(self, [self.g_train_adversarial, self.g_loss1, self.g_loss2], self.batch_size)
                else:
                    _, gen_loss1 = training_step(self, [self.g_train_MLE, -self.mle_lossG], self.batch_size)
                for i in range(disc_itr):
                    _, disc_loss1, disc_loss2 = training_step(self, [self.d_train_adversarial, self.d_loss1, self.d_loss2], disc_batch)

                self.Gen_loss1, self.Gen_loss2, self.Disc_loss1, self.Disc_loss2 = \
                    (self.Gen_loss1+gen_loss1, self.Gen_loss2+gen_loss2, self.Disc_loss1+disc_loss1, self.Disc_loss2+disc_loss2)

                if (math.isnan(gen_loss1)) or (math.isnan(gen_loss2)) or (math.isnan(disc_loss1)) or (math.isnan(disc_loss2)):
                    print("")
                    print("Loss is nan")
                    print("")
                    cont = False
                    self.save_data(step)
                    
                if ((step > self.model_params.min_steps) and (early_stopping(self.dataG[0], 4) or early_stopping(self.dataG[2], 4))) or \
                    (step>self.model_params.max_steps):
                    cont = False
                
                self.save_data(step)
                testing_step(self, step)
                calculate_auc_discriminator(self, step)
                step += 1
            
            except KeyboardInterrupt:
                cont = False

        self.save_data(step)
        tf.reset_default_graph()

    def save_data(self, step):
        if (step%self.model_params.saving_step==0):
            np.save(self.model_params.name+"_disc", np.array(self.dataD))
            np.save(self.model_params.name+"_gen", np.array(self.dataG))
            np.save(self.model_params.name+"_gen_emb"+str(step), self._sess.run(self.generator.embeddings_tensorflow))