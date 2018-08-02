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

from gen_cnn import Gen_CNN_Model
from lstm import LSTM_Model
from word2vec import W2V_Model
from word2vec_google import *

class Self_Basket_Completion_Model(object):

    def __init__(self, model):
        self.model_params = model
        self.train_data, self.test_data, self.X_train, self.Y_train, self.X_test, self.Y_test = list(), list(), list(), list(), list(), list()
        self.LSTM_labels_train, self.LSTM_labels_test = list(), list()
        self.printing_step = 500
        self.neg_sampled_pretraining = model.neg_sampled

        self.training_data, self.test_data, self.test_list_batches = model.training_data, model.test_data, model.test_list_batches
        self.num_epochs, self.batch_size, self.vocabulary_size, self.popularity_distribution = model.epoch, model.batch_size, model.vocabulary_size, model.popularity_distribution
        self.seq_length, self.epoch = model.seq_length, model.epoch
        self.embedding_size, self.embedding_matrix, self.use_pretrained_embeddings = model.embedding_size, model.embedding_matrix, model.use_pretrained_embeddings
        self.adv_generator_loss, self.adv_discriminator_loss = model.adv_generator_loss, model.adv_discriminator_loss
        self.adv_negD, self.random_negD = model.negD
        self.discriminator_type = model.D_type
        self.discriminator_samples_type = model.discriminator_samples_type
        self.one_guy_sample = np.random.choice(self.vocabulary_size-1)
        self.MPRD, self.confintD, self.p1D, self.confintp1D, self.p1pD, self.confintp1pD, self.true_neg_prop, self.auc_score = \
            list(), list(), list(), list(), list(), list(), list(), list()

    def create_graph(self):
        self.train_words, self.label_words, self.dropout, self.popularity_distribution_tensor, self.context_words = \
            create_placeholders(self)
        create_discriminator(self)
        self.before_softmax_G = self.before_softmax_D
        self.d_loss2 = -discriminator_adversarial_loss(self)
        self.d_loss = 0.5*(self.d_loss1+self.d_loss2) if (self.adv_discriminator_loss[1]=="Mixed") else self.d_loss2
        self.disc_optimizer, self.disc_optimizer_adv, self.disc_optimizer_softmax = tf.train.AdamOptimizer(1e-3, beta1=0.8 ,beta2= 0.9, epsilon=1e-5), tf.train.AdamOptimizer(1e-3, beta1=0.8 ,beta2= 0.9, epsilon=1e-5), tf.train.AdamOptimizer(1e-3, beta1=0.8 ,beta2= 0.9, epsilon=1e-5)
        self.d_train_normal, self.d_train_adversarial, self.d_softmax = self.disc_optimizer.minimize(self.d_loss1, var_list=self.d_weights), self.disc_optimizer_softmax.minimize(self.d_loss, var_list=self.d_weights), self.disc_optimizer_adv.minimize(self.softmax_loss, var_list=self.d_weights)
        self.softmax_grad, self.adv_grad = tf.gradients(self.softmax_loss, self.d_weights), tf.gradients(self.d_loss, self.d_weights)

    def train_model_with_tensorflow(self):
        self.create_graph()
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())
        step, cont = 0, True
        self.Gen_loss1, self.Gen_loss2, self.Disc_loss1, self.Disc_loss2, self.pic_number = 0, 0, 0, 0, 0
        disc_loss1, disc_loss2 = 0, 0

        while cont:

            if (self.model_params.model_type=="baseline"):
                _, disc_loss1 = training_step(self, list_of_operations_to_run=[self.d_train_normal, self.d_loss1])
            elif (self.model_params.model_type=="softmax"):
                _, disc_loss1 = training_step(self, list_of_operations_to_run=[self.d_softmax, self.softmax_loss])
            else:
                _, disc_loss1, disc_loss2 = training_step(self, list_of_operations_to_run=[self.d_train_adversarial, self.d_loss1, self.d_loss2])

            if (step % self.printing_step == 0) and (step>0):
                print_gen_and_disc_losses(self, step)

            if (step % self.printing_step == 0):
                if (self.model_params.metric=="MPR"):
                    print_confidence_intervals(self, "selfplay")

                if (self.model_params.type_of_data=="synthetic"):
                    true_neg_prop, auc_score = print_candidate_sampling(self, [self.model_params.model_type, "DISC"])
                    self.true_neg_prop.append(true_neg_prop)
                    self.auc_score.append(auc_score)

                if (self.model_params.metric=="Analogy"):
                    check_analogy(self)
                    np.save("nlp/"+self.model_params.name+"_embeddings", self._sess.run([self.discriminator.embeddings_tensorflow])[0])

            step, self.Disc_loss1, self.Disc_loss2 = (step+1, self.Disc_loss1+disc_loss1, self.Disc_loss2+disc_loss2)

            if (step > self.model_params.min_steps) and (self.MPRD[-1] < np.mean(self.MPRD[-10:-1])):
                if (self.model_params.model_type == "softmax"):
                    data = np.array([self.MPRD, self.confintD, self.p1D, self.confintp1D, self.p1pD, self.confintp1pD, self.true_neg_prop, self.auc_score])
                    print(data)
                    np.save(self.model_params.name, data)
                    self.model_params.model_type, self.model_params.name = "selfplay", "softmax_selfplay" + "_" + self.model_params.dataset +"_"+ self.model_params.G_type + "_" + str(self.model_params.neg_sampled)
                else:
                    cont = False

        data = np.array([self.MPRD, self.confintD, self.p1D, self.confintp1D, self.p1pD, self.confintp1pD, self.true_neg_prop, self.auc_score])
        print(data)
        if (self.model_params.metric=="MPR") or (self.model_params.metric=="AUC"):
            np.save(self.model_params.name, data)
