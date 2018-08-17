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
        self.seed = 1234
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

        self.max_basket_size, self.test_data_max_size = 50, 5000
        self.dataset = dataset
        self.data, self.folder, self.metric, self.type_of_data = load_data(self)
        self.vocabulary_size, self.vocabulary_size2 = get_vocabulary_size(self)
        process_and_shuffle_data(self)

        self.proportion_training, self.proportion_test = (0.8, 0.2) if (self.dataset != "text8") else (1.0, 0)
        self.num_training_instances, self.num_test_instances = int(len(self.data)*self.proportion_training), int(len(self.data)*self.proportion_test)
        self.test_data, self.training_data = self.data[:self.num_test_instances], self.data[self.num_test_instances:]
        self.test_list_batches, self.test_baskets, self.test_items = get_test_list_batches(self)

        self.popularity_distribution = 1 #np.array(get_popularity_dist(self.training_data, self.vocabulary_size))
        self.embedding_size, self.batch_size, self.epoch, self.speeding_factor, self.seq_length = 128, 200, 10, 25, 25
        if (self.type_of_data=="synthetic") or (self.dataset == "text8"):
            self.seq_length= 1

        self.use_pretrained_embeddings=False
        print(self.dataset, self.metric, self.type_of_data)
        print("")
        print("Length data, train and test ", len(self.data), len(self.training_data), len(self.test_data))            
        print("Vocabulary sizes", self.vocabulary_size, self.vocabulary_size2)
        print("")

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

def create_all_attributes(self, dataset, model_type, negD, negG, min_steps, neg_sampled, G_type, D_type, sampling_type):
        self.embedding_matrix = np.load("embedding_matrix.npy") if (os.path.isfile("embedding_matrix.npy")) else np.zeros((1,1))
        self.use_pretrained_embeddings = False
        self.G_type, self.D_type = G_type, D_type
        self.model_type = model_type
        if (model_type=="AIS"):
            self.adv_generator_loss, self.adv_discriminator_loss = ["ADV_IS", "Not_Mixed"], ["SS", "Not_Mixed"]
            self.negD = [5, 1] if (neg_sampled==1) else [25, 1]
        elif (model_type=="MALIGAN"):
            self.adv_generator_loss, self.adv_discriminator_loss = ["MALIGAN", "Mixed"], ["SS", "Not_Mixed"]
            self.negD = [5, 1] if (neg_sampled==1) else [25, 1]
        elif (model_type=="softmax"):
            self.adv_generator_loss, self.adv_discriminator_loss = ["Random_ADV_IS", "Not_Mixed"], ["softmax", "Not_Mixed"]
            self.negD = negD
        elif (model_type=="MLE"):
            self.adv_generator_loss, self.adv_discriminator_loss = ["Random_ADV_IS", "Not_Mixed"], ["MLE", "Not_Mixed"]
            self.negD = negD
        elif ("BCE" in model_type):
            self.adv_generator_loss, self.adv_discriminator_loss = ["Random_ADV_IS", "Not_Mixed"], ["BCE", "Not_Mixed"]
            self.negD = negD
        elif (model_type=="selfplay"):
             self.adv_generator_loss, self.adv_discriminator_loss = ["Random_ADV_IS", "Not_Mixed"], ["SS", "Not_Mixed"]
             self.negD = [1, 0] if (neg_sampled<1) else [neg_sampled, 0]
        else:
            self.adv_generator_loss, self.adv_discriminator_loss = ["Random_ADV_IS", "Not_Mixed"], ["SS", "Not_Mixed"]
            self.negD = [0, 1] if (neg_sampled<1) else [0, neg_sampled]
        self.negG = [1, 0] if (neg_sampled<1) else [neg_sampled, 0]
        self.neg_sampled = neg_sampled
        self.name = model_type + "_" + dataset +"_"+ G_type + "_" + str(neg_sampled)
        self.name = self.name +"_"+sampling_type if (sampling_type=="no_stochastic") else self.name
        self.discriminator_samples_type, self.generator_samples_type = [sampling_type, "uniform"], [sampling_type, "uniform"]
        self.min_steps = min_steps

class Basket_Generation(Model):
    def __init__(self, dataset, model_type, negD, negG, min_steps, neg_sampled, G_type, D_type):
        Model.__init__(self, dataset)
        create_all_attributes(self, dataset, model_type, negD, negG, min_steps, neg_sampled, G_type, D_type, sampling_type)
        gan = Basket_Generation_Model(self)
        gan.train_model_with_tensorflow()

class Basket_Completion(Model):
    def __init__(self, dataset, model_type, negD=[1,0], negG=[1,0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v", sampling_type="stochastic"):
        Model.__init__(self, dataset)
        create_all_attributes(self, dataset, model_type, negD, negG, min_steps, neg_sampled, G_type, D_type, sampling_type)
        gan = Basket_Completion_Model(self)
        gan.train_model_with_tensorflow()

class Self_Basket_Completion(Model):
    def __init__(self, dataset, model_type, negD=[1,0], negG=[1,0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v", sampling_type="stochastic"):
        Model.__init__(self, dataset)
        create_all_attributes(self, dataset, model_type, negD, negG, min_steps, neg_sampled, G_type, D_type, sampling_type)
        self_play = Self_Basket_Completion_Model(self)
        self_play.train_model_with_tensorflow()

def get_class(model):
    if (model=="AIS") or (model=="MALIGAN"):
        return Basket_Completion
    else:
        return Self_Basket_Completion

def get_min_steps(dataset):
    return 2500 if (dataset=="Amazon") else 20000

def test_all_softmax(list_of_datasets):
    for dataset in list_of_datasets:
        Self_Basket_Completion(dataset=dataset, min_steps=get_min_steps(dataset), model_type="softmax", G_type="w2v", D_type="w2v")
        Self_Basket_Completion(dataset=dataset, min_steps=get_min_steps(dataset), model_type="softmax", G_type="CNN", D_type="CNN")

def test_all_MLE(list_of_datasets):
    for dataset in list_of_datasets:
        Self_Basket_Completion(dataset=dataset, min_steps=get_min_steps(dataset), model_type="MLE", G_type="w2v", D_type="w2v")
        Self_Basket_Completion(dataset=dataset, min_steps=get_min_steps(dataset), model_type="MLE", G_type="CNN", D_type="CNN")

def test_other_models(list_of_models, list_of_datasets, list_of_NS, list_of_networks, sampling_type="stochastic"):
    #id_to_dataset = {1:"Amazon", 2:"Belgian", 3:"UK", 4:"movielens", 5:"netflix"}
    for net in list_of_networks:
        for model in list_of_models:
            for dataset in list_of_datasets:
                for NS in list_of_NS:
                    get_class(model)(dataset=dataset, model_type=model, min_steps=get_min_steps(dataset), sampling_type=sampling_type, \
                        neg_sampled=NS, G_type=net, D_type=net)
                        
test_all_softmax(["Amazon", "Belgian", "UK", "movielens", "netflix"])
test_all_MLE(["Amazon", "Belgian", "UK", "movielens", "netflix"])
test_other_models(["uniform", "baseline"], ["Amazon"], [0.5, 0.75, 1, 2, 5, 25, 50], ["w2v"])
test_other_models(["selfplay", "AIS", "MALIGAN"], ["Amazon"], [0.5, 0.75, 1, 2, 5, 25, 50], ["w2v"])
test_other_models(["uniform", "baseline"], ["Amazon"], [0.5, 0.75, 1, 2, 5, 25, 50], ["CNN"])
test_other_models(["selfplay", "AIS", "MALIGAN"], ["Amazon"], [0.5, 0.75, 1, 2, 5, 25, 50], ["CNN"])


test_other_models(["uniform", "baseline"], ["Belgian"], [0.5,0.75, 1, 2, 5, 25, 50], ["w2v"])
test_other_models(["selfplay", "AIS", "MALIGAN"], ["Belgian"], [0.5,0.75, 1, 2, 5, 25, 50], ["w2v"])
test_other_models(["uniform", "baseline"], ["netflix"], [0.5, 0.75, 1, 2, 5, 25, 50], ["w2v"])
test_other_models(["selfplay", "AIS", "MALIGAN"], ["netflix"], [0.5, 0.75, 1, 2, 5, 25, 50], ["w2v"])

test_other_models(["uniform", "baseline"], ["Belgian"], [0.5,0.75, 1, 2, 5, 25, 50], ["CNN"])
test_other_models(["selfplay", "AIS", "MALIGAN"], ["Belgian"], [0.5,0.75, 1, 2, 5, 25, 50], ["CNN"])
test_other_models(["uniform", "baseline"], ["netflix"], [0.5, 0.75, 1, 2, 5, 25, 50], ["CNN"])
test_other_models(["selfplay", "AIS", "MALIGAN"], ["netflix"], [0.5, 0.75, 1, 2, 5, 25, 50], ["CNN"])


test_other_models(["uniform", "baseline"], ["UK"], [0.5, 0.75, 1, 2, 5, 25, 50], ["w2v"])
test_other_models(["selfplay", "AIS", "MALIGAN"], ["UK"], [0.5, 0.75, 1, 2, 5, 25, 50], ["w2v"])
test_other_models(["uniform", "baseline"], ["movielens"], [0.5, 0.75, 1, 2, 5, 25, 50], ["w2v"])
test_other_models(["selfplay", "AIS", "MALIGAN"], ["movielens"], [0.5, 0.75, 1, 2, 5, 25, 50], ["w2v"])

test_other_models(["uniform", "baseline"], ["UK"], [0.5, 0.75, 1, 2, 5, 25, 50], ["CNN"])
test_other_models(["selfplay", "AIS", "MALIGAN"], ["UK"], [0.5, 0.75, 1, 2, 5, 25, 50], ["CNN"])
test_other_models(["uniform", "baseline"], ["movielens"], [0.5, 0.75, 1, 2, 5, 25, 50], ["CNN"])
test_other_models(["selfplay", "AIS", "MALIGAN"], ["movielens"], [0.5, 0.75, 1, 2, 5, 25, 50], ["CNN"])






                                                                                                          