import matplotlib
matplotlib.use('Agg')
import os
import itertools
import copy
import numpy as np
from tqdm import tqdm
from collections import Counter
from scipy.spatial.distance import cosine
from random import shuffle
import random
import math
import tensorflow as tf

from tools import *
from tools_read_data import *

#from lstm import LSTM_Model
#from cbow import CBOW_Model
from gen_cnn import Gen_CNN_Model
from word2vec import W2V_Model
from metric_utils import print_results_predictions
from self_basket_completion import Self_Basket_Completion_Model
from basket_completion import Basket_Completion_Model
#from basket_generation import Basket_Generation_Model


class Model(object):
    def __init__(self, dataset):
        random.seed(1234)
        np.random.seed(1234)
        tf.set_random_seed(1234)

        self.max_basket_size, self.test_data_max_size = 50, 5000
        self.dataset = dataset
        self.data, self.folder, self.metric, self.type_of_data = load_data(self)
        self.vocabulary_size = get_vocabulary_size(self)
        process_and_shuffle_data(self)

        self.proportion_training, self.proportion_test = (0.8, 0.2) if (self.dataset != "text8") else (1.0, 0)
        self.num_training_instances, self.num_test_instances = int(len(self.data)*self.proportion_training), int(len(self.data)*self.proportion_test)
        self.test_data, self.training_data = self.data[:self.num_test_instances], self.data[self.num_test_instances:]
        self.test_list_batches, self.test_baskets, self.test_items = get_test_list_batches(self)

        self.popularity_distribution = np.array(get_popularity_dist(self.training_data, self.vocabulary_size))
        self.embedding_size, self.batch_size, self.epoch, self.speeding_factor, self.seq_length = 128, 200, 10, 25, 25
        if (self.type_of_data=="synthetic") or (self.dataset == "text8"):
            self.seq_length= 1

        self.use_pretrained_embeddings=False
        print("Length data, train and test ", len(self.data), len(self.training_data), len(self.test_data))            
        print("Vocabulary size", self.vocabulary_size)

class w2v(Model):
    def __init__(self):
        Model.__init__(self)
        embedding_matrix = np.load("embedding_matrix.npy") if (os.path.isfile("embedding_matrix.npy")) else np.zeros((1,1))
        self.embedding_matrix=embedding_matrix
        w2v = W2V_Model(self)
        w2v.train_model_with_tensorflow()

class CBOW(Model):
    def __init__(self):
        Model.__init__(self)
        embedding_matrix = np.load("embedding_matrix.npy") if (os.path.isfile("embedding_matrix.npy")) else np.zeros((1,1))
        self.embedding_matrix=embedding_matrix
        w2v = CBOW_Model(self)
        w2v.training_data, w2v.test_data = self.training_data, self.test_data
        w2v.X_train, _, w2v.Y_train = preprocessing_data(self.training_data, self.seq_length, 'cnn')
        w2v.X_test, _, w2v.Y_test = preprocessing_data_one_shuffle(self.test_data, self.seq_length, 'cnn') #no shuffle for test
        w2v.train_model_with_tensorflow()

class LSTM(Model):
    def __init__(self):
        Model.__init__(self)
        embedding_matrix = np.vstack((np.zeros((1,64)),np.load("embedding_matrix.npy"))) if (os.path.isfile("embedding_matrix.npy")) else np.zeros((1,1))
        self.embedding_matrix=embedding_matrix
        lstm = LSTM_Model(self)
        lstm.X_train, lstm.LSTM_labels_train, lstm.Y_train = preprocessing_data(self.training_data, self.seq_length, 'lstm')
        lstm.X_test, lstm.LSTM_labels_test, lstm.Y_test = preprocessing_data_one_shuffle(self.test_data, self.seq_length, 'lstm')
        lstm.train_model_with_tensorflow()

class CNN(Model):
    def __init__(self, type_of_data, dataset, min_steps):
        Model.__init__(self, type_of_data, dataset)
        embedding_matrix = np.load("embedding_matrix.npy") if (os.path.isfile("embedding_matrix.npy")) else np.zeros((1,1))
        self.embedding_matrix = embedding_matrix
        self.min_steps = min_steps
        cnn = Gen_CNN_Model(self)
        cnn.initialize_model()
        
        while (cnn.step < self.number_of_training_step):
            cnn.training_step()

        data = np.array([MPRs, precision1s, precision1ps])
        np.save("textCNN_results", data)
        cnn.train_model_with_tensorflow()

def create_all_attributes(self, dataset, model_type, negD, negG, min_steps, neg_sampled, G_type, D_type):
        self.embedding_matrix = np.load("embedding_matrix.npy") if (os.path.isfile("embedding_matrix.npy")) else np.zeros((1,1))
        self.use_pretrained_embeddings = False
        self.G_type, self.D_type = G_type, D_type
        self.model_type = model_type
        if (model_type=="AIS"):
            self.adv_generator_loss, self.adv_discriminator_loss = ["ADV_IS", "Not_Mixed"], ["SS", "Not_Mixed"]
        elif (model_type=="softmax"):
            self.adv_generator_loss, self.adv_discriminator_loss = ["ADV_IS", "Not_Mixed"], ["softmax", "Not_Mixed"]
        elif (model_type=="MALIGAN"):
            self.adv_generator_loss, self.adv_discriminator_loss = ["MALIGAN", "Mixed"], ["SS", "Not_Mixed"]
        elif ("BCE" in model_type):
            self.adv_generator_loss, self.adv_discriminator_loss = ["Random_ADV_IS", "Not_Mixed"], ["BCE", "Not_Mixed"]
        else:
             self.adv_generator_loss, self.adv_discriminator_loss = ["Random_ADV_IS", "Not_Mixed"], ["SS", "Not_Mixed"]
        self.neg_sampled = neg_sampled
        self.negD, self.negG = negD, negG
        self.name = model_type + "_" + dataset +"_"+ G_type + "_" + str(neg_sampled)
        self.discriminator_samples_type = ["stochastic", "uniform"]
        self.min_steps = min_steps

class Basket_Generation(Model):
    def __init__(self, dataset, model_type, negD, negG, min_steps, neg_sampled, G_type, D_type):
        Model.__init__(self, dataset)
        create_all_attributes(self, dataset, model_type, negD, negG, min_steps, neg_sampled, G_type, D_type)
        gan = Basket_Generation_Model(self)
        gan.train_model_with_tensorflow()

class Basket_Completion(Model):
    def __init__(self, dataset, model_type, negD, negG, min_steps, neg_sampled, G_type, D_type):
        Model.__init__(self, dataset)
        create_all_attributes(self, dataset, model_type, negD, negG, min_steps, neg_sampled, G_type, D_type)
        gan = Basket_Completion_Model(self)
        gan.train_model_with_tensorflow()

class Self_Basket_Completion(Model):
    def __init__(self, dataset, model_type, negD, negG, min_steps, neg_sampled, G_type, D_type):
        Model.__init__(self, dataset)
        create_all_attributes(self, dataset, model_type, negD, negG, min_steps, neg_sampled, G_type, D_type)
        self_play = Self_Basket_Completion_Model(self)
        self_play.train_model_with_tensorflow()

def launch(dataset, model_type, negD, negG, min_steps, neg_sampled, G_type, D_type):
    if (model_type=="AIS") or (model_type=="MALIGAN"):
        Basket_Completion(dataset, model_type, negD, negG, min_steps, neg_sampled, G_type, D_type)
    else:
        Self_Basket_Completion(dataset, model_type, negD, negG, min_steps, neg_sampled, G_type, D_type)


base = launch(dataset="Amazon", model_type="softmax", negD=[0, 1], negG=[25, 0], min_steps=0, neg_sampled=25, G_type="w2v", D_type="w2v")

###############################1
#base = Self_Basket_Completion(dataset="Amazon", model_type="baseline", negD=[5, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v")
#unif = Self_Basket_Completion(dataset="Amazon", model_type="uniform", negD=[0, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v")
#unifBCE = Self_Basket_Completion(dataset="Amazon", model_type="uniformBCE", negD=[0, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v")
#self = Self_Basket_Completion(dataset="Amazon", model_type="selfplay", negD=[1, 0], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v")
#selfBCE = Self_Basket_Completion(dataset="Amazon", model_type="selfplayBCE", negD=[1, 0], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v")
#ais = Basket_Completion(dataset="Amazon", model_type="AIS", negD=[5, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v")
#maligan = Basket_Completion(dataset="Amazon", model_type="MALIGAN", negD=[5, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v")

#base = Self_Basket_Completion(dataset="Amazon", model_type="baseline", negD=[5, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="CNN", D_type="CNN")
#unif = Self_Basket_Completion(dataset="Amazon", model_type="uniform", negD=[0, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="CNN", D_type="CNN")
#unifBCE = Self_Basket_Completion(dataset="Amazon", model_type="uniformBCE", negD=[0, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="CNN", D_type="CNN")
#self = Self_Basket_Completion(dataset="Amazon", model_type="selfplay", negD=[1, 0], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="CNN", D_type="CNN")
#selfBCE = Self_Basket_Completion(dataset="Amazon", model_type="selfplayBCE", negD=[1, 0], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="CNN", D_type="CNN")
#ais = Basket_Completion(dataset="Amazon", model_type="AIS", negD=[5, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="CNN", D_type="CNN")
#maligan = Basket_Completion(dataset="Amazon", model_type="MALIGAN", negD=[5, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="CNN", D_type="CNN")

###############################2
#base = Self_Basket_Completion(dataset="Belgian", model_type="baseline", negD=[5, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v")
#unif = Self_Basket_Completion(dataset="Belgian", model_type="uniform", negD=[0, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v")
#unifBCE = Self_Basket_Completion(dataset="Belgian", model_type="uniformBCE", negD=[0, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v")
#self = Self_Basket_Completion(dataset="Belgian", model_type="selfplay", negD=[1, 0], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v")
#selfBCE = Self_Basket_Completion(dataset="Belgian", model_type="selfplayBCE", negD=[1, 0], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v")
#ais = Basket_Completion(dataset="Belgian", model_type="AIS", negD=[5, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v")
#maligan = Basket_Completion(dataset="Belgian", model_type="MALIGAN", negD=[5, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v")

#base = Self_Basket_Completion(dataset="Belgian", model_type="baseline", negD=[5, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="CNN", D_type="CNN")
#unif = Self_Basket_Completion(dataset="Belgian", model_type="uniform", negD=[0, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="CNN", D_type="CNN")
#unifBCE = Self_Basket_Completion(dataset="Belgian", model_type="uniformBCE", negD=[0, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="CNN", D_type="CNN")
#self = Self_Basket_Completion(dataset="Belgian", model_type="selfplay", negD=[1, 0], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="CNN", D_type="CNN")
#selfBCE = Self_Basket_Completion(dataset="Belgian", model_type="selfplayBCE", negD=[1, 0], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="CNN", D_type="CNN")
#ais = Basket_Completion(dataset="Belgian", model_type="AIS", negD=[5, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="CNN", D_type="CNN")
#maligan = Basket_Completion(dataset="Belgian", model_type="MALIGAN", negD=[5, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="CNN", D_type="CNN")

###############################3
#base = Self_Basket_Completion(dataset="UK", model_type="baseline", negD=[5, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v")
#unif = Self_Basket_Completion(dataset="UK", model_type="uniform", negD=[0, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v")
#unifBCE = Self_Basket_Completion(dataset="UK", model_type="uniformBCE", negD=[0, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v")
#self = Self_Basket_Completion(dataset="UK", model_type="selfplay", negD=[1, 0], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v")
#selfBCE = Self_Basket_Completion(dataset="UK", model_type="selfplayBCE", negD=[1, 0], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v")
#ais = Basket_Completion(dataset="UK", model_type="AIS", negD=[5, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v")
#maligan = Basket_Completion(dataset="UK", model_type="MALIGAN", negD=[5, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v")

#base = Self_Basket_Completion(dataset="UK", model_type="baseline", negD=[5, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="CNN", D_type="CNN")
#unif = Self_Basket_Completion(dataset="UK", model_type="uniform", negD=[0, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="CNN", D_type="CNN")
#unifBCE = Self_Basket_Completion(dataset="UK", model_type="uniformBCE", negD=[0, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="CNN", D_type="CNN")
#self = Self_Basket_Completion(dataset="UK", model_type="selfplay", negD=[1, 0], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="CNN", D_type="CNN")
#selfBCE = Self_Basket_Completion(dataset="UK", model_type="selfplayBCE", negD=[1, 0], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="CNN", D_type="CNN")
#ais = Basket_Completion(dataset="UK", model_type="AIS", negD=[5, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="CNN", D_type="CNN")
#maligan = Basket_Completion(dataset="UK", model_type="MALIGAN", negD=[5, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="CNN", D_type="CNN")

###############################4
#base = Self_Basket_Completion(dataset="movielens", model_type="baseline", negD=[5, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v")
#unif = Self_Basket_Completion(dataset="movielens", model_type="uniform", negD=[0, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v")
#unifBCE = Self_Basket_Completion(dataset="movielens", model_type="uniformBCE", negD=[0, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v")
#self = Self_Basket_Completion(dataset="movielens", model_type="selfplay", negD=[1, 0], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v")
#selfBCE = Self_Basket_Completion(dataset="movielens", model_type="selfplayBCE", negD=[1, 0], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v")
#ais = Basket_Completion(dataset="movielens", model_type="AIS", negD=[5, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v")
#maligan = Basket_Completion(dataset="movielens", model_type="MALIGAN", negD=[5, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v")

#base = Self_Basket_Completion(dataset="movielens", model_type="baseline", negD=[5, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="CNN", D_type="CNN")
#unif = Self_Basket_Completion(dataset="movielens", model_type="uniform", negD=[0, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="CNN", D_type="CNN")
#unifBCE = Self_Basket_Completion(dataset="movielens", model_type="uniformBCE", negD=[0, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="CNN", D_type="CNN")
#self = Self_Basket_Completion(dataset="movielens", model_type="selfplay", negD=[1, 0], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="CNN", D_type="CNN")
#selfBCE = Self_Basket_Completion(dataset="movielens", model_type="selfplayBCE", negD=[1, 0], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="CNN", D_type="CNN")
#ais = Basket_Completion(dataset="movielens", model_type="AIS", negD=[5, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="CNN", D_type="CNN")
#maligan = Basket_Completion(dataset="movielens", model_type="MALIGAN", negD=[5, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="CNN", D_type="CNN")

###############################5
#base = Self_Basket_Completion(dataset="netflix", model_type="baseline", negD=[5, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v")
#unif = Self_Basket_Completion(dataset="netflix", model_type="uniform", negD=[0, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v")
#unifBCE = Self_Basket_Completion(dataset="netflix", model_type="uniformBCE", negD=[0, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v")
#self = Self_Basket_Completion(dataset="netflix", model_type="selfplay", negD=[1, 0], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v")
#selfBCE = Self_Basket_Completion(dataset="netflix", model_type="selfplayBCE", negD=[1, 0], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v")
#ais = Basket_Completion(dataset="netflix", model_type="AIS", negD=[5, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v")
#maligan = Basket_Completion(dataset="netflix", model_type="MALIGAN", negD=[5, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v")

#base = Self_Basket_Completion(dataset="netflix", model_type="baseline", negD=[5, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="CNN", D_type="CNN")
#unif = Self_Basket_Completion(dataset="netflix", model_type="uniform", negD=[0, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="CNN", D_type="CNN")
#unifBCE = Self_Basket_Completion(dataset="netflix", model_type="uniformBCE", negD=[0, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="CNN", D_type="CNN")
#self = Self_Basket_Completion(dataset="netflix", model_type="selfplay", negD=[1, 0], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="CNN", D_type="CNN")
#selfBCE = Self_Basket_Completion(dataset="netflix", model_type="selfplayBCE", negD=[1, 0], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="CNN", D_type="CNN")
#ais = Basket_Completion(dataset="netflix", model_type="AIS", negD=[5, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="CNN", D_type="CNN")
#maligan = Basket_Completion(dataset="netflix", model_type="MALIGAN", negD=[5, 1], negG=[1, 0], min_steps=2500, neg_sampled=1, G_type="CNN", D_type="CNN")