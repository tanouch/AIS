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
import concurrent.futures

from tools import *
from tools_read_data import *
from gen_cnn import Gen_CNN_Model
from word2vec import W2V_Model
from metric_utils import print_results_predictions, calculate_mpr_and_auc_pairs
from self_basket_completion import Self_Basket_Completion_Model
from basket_completion import Basket_Completion_Model


class Model(object):
    def __init__(self, dataset, task_mode):
        self.seed = 1234
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)
        self.rnd = np.random.RandomState(1234)

        self.max_basket_size, self.test_data_max_size = 50, 5000
        self.dataset, self.task_mode, self.working_with_pairs = dataset, task_mode, True
        self.data, self.folder, self.metric, self.type_of_data, self.dictionnary = load_data(self)
        self.dict_co_occurences = get_co_occurences_dict(self.data, self.task_mode)
        
        if task_mode=="item-item":
            self.data = create_item_to_item_data(dataset, self.data, self.rnd)
        if task_mode=="user-item":
            self.data = create_user_to_item_data(dataset, self.data, self.rnd)

        self.vocabulary_size, self.vocabulary_size2 = get_vocabulary_size(self)
        self.test_data, self.training_data = split_data(self.data)
        self.test_list_batches, self.test_baskets, self.test_items = get_test_list_batches(self)
        self.check_words_similarity = np.array(list(np.random.choice(100, 25, replace=False)) \
            + list(500 + np.random.choice(500, 25, replace=False)))
        
        self.embedding_size, self.batch_size, self.epoch, self.speeding_factor, self.seq_length = 150, 200, 10, 25, 3
        self.conv_filters = [2, 4, 6, 8, 12]
        if (self.type_of_data=="synthetic") or (self.dataset == "text8"):
            self.seq_length= 1

        self.use_pretrained_embeddings=False
        print("")
        print("Length data, train and test ", len(self.data), len(self.training_data), len(self.test_data))            
        print("Vocabulary sizes", self.vocabulary_size, self.vocabulary_size2)
        print("")


class CNN(Model):
    def __init__(self, dataset, task_mode):
        Model.__init__(self, dataset, task_mode)
        cnn = Gen_CNN_Model(self)
        cnn.initialize_model()
        
        while (cnn.step < self.number_of_training_step):
            cnn.training_step()

        data = np.array([MPRs, precision1s, precision1ps])
        np.save("textCNN_results", data)
        cnn.train_model_with_tensorflow()

def create_all_attributes(self, dataset, model_type, negD, negG, min_steps, neg_sampled, G_type, D_type, sampling_type):
        self.use_pretrained_embeddings, self.embedding_matrix = False, 0
        self.G_type, self.D_type = G_type, D_type
        self.model_type = model_type
        self.negG = [1, 0] if (neg_sampled<1) else [neg_sampled, 0]
        self.negD = [1, 0]
        if (model_type=="AIS"):
            self.adv_generator_loss, self.adv_discriminator_loss = ["AIS", "Not_Mixed"], ["SS", "Not_Mixed"]
            self.negD = [neg_sampled, 0]

        elif (model_type=="AIS-BCE"):
            self.adv_generator_loss, self.adv_discriminator_loss = ["AIS-BCE", "Not_Mixed"], ["BCE", "Not_Mixed"]
            self.negD = [neg_sampled, 0]
        
        elif (model_type=="MALIGAN"):
            self.adv_generator_loss, self.adv_discriminator_loss = ["MALIGAN", "Mixed"], ["SS", "Not_Mixed"]
            self.negD = [neg_sampled, 0]
        
        elif (model_type=="softmax"):
            self.adv_generator_loss, self.adv_discriminator_loss = ["AIS", "Not_Mixed"], ["softmax", "Not_Mixed"]
        
        elif (model_type=="MLE"):
            self.adv_generator_loss, self.adv_discriminator_loss = ["AIS", "Not_Mixed"], ["MLE", "Not_Mixed"]
        
        elif ("BCE" in model_type):
            self.adv_generator_loss, self.adv_discriminator_loss = ["AIS", "Not_Mixed"], ["BCE", "Not_Mixed"]
        
        elif (model_type=="selfplay"):
             self.adv_generator_loss, self.adv_discriminator_loss = ["AIS", "Not_Mixed"], ["SS", "Not_Mixed"]
             self.negD = [1, 0] if (neg_sampled<1) else [neg_sampled, 0]
        
        else:
            self.adv_generator_loss, self.adv_discriminator_loss = ["AIS", "Not_Mixed"], ["SS", "Not_Mixed"]
            self.negD = [0, 1] if (neg_sampled<1) else [0, neg_sampled]
        
        self.neg_sampled = neg_sampled
        self.name = model_type + "_" + dataset +"_"+ G_type + "_" + str(neg_sampled)
        self.name = self.name +"_"+sampling_type if (sampling_type=="no_stochastic") else self.name
        self.discriminator_samples_type, self.generator_samples_type = [sampling_type, "uniform"], [sampling_type, "uniform"]
        self.min_steps = min_steps
        self.printing_step = 5000
        
        print(dataset, model_type, neg_sampled, "negD", self.negD, "negG", self.negG, "G_type", G_type, "D_type", D_type)

class Basket_Completion(Model):
    def __init__(self, dataset, task_mode, model_type, negD=[1,0], negG=[1,0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v", sampling_type="stochastic"):
        Model.__init__(self, dataset, task_mode)
        create_all_attributes(self, dataset, model_type, negD, negG, min_steps, neg_sampled, G_type, D_type, sampling_type)
        gan = Basket_Completion_Model(self)
        gan.train_model_with_tensorflow()
        del gan

class Self_Basket_Completion(Model):
    def __init__(self, dataset, task_mode, model_type, negD=[1,0], negG=[1,0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v", sampling_type="stochastic"):
        Model.__init__(self, dataset, task_mode)
        create_all_attributes(self, dataset, model_type, negD, negG, min_steps, neg_sampled, G_type, D_type, sampling_type)
        self_play = Self_Basket_Completion_Model(self)
        self_play.train_model_with_tensorflow()
        del self_play

class Pre_Loaded_Embedding(Model):
    def __init__(self, dataset, task_mode, model_type, negD=[1,0], negG=[1,0], min_steps=2500, neg_sampled=1, G_type="w2v", D_type="w2v", sampling_type="stochastic"):
        Model.__init__(self, dataset, task_mode)
        create_all_attributes(self, dataset, model_type, negD, negG, min_steps, neg_sampled, G_type, D_type, sampling_type)
        self.use_pretrained_embeddings = True
        #self.user_embeddings, self.items_embeddings = np.load("ivan_embeddings/belg_users_d=300.npy")[:,-1:], np.load("ivan_embeddings/belg_items_d=300.npy")[:,-1:]
        self.user_embeddings = np.ones((self.vocabulary_size,1))
        self.items_embeddings = np.zeros((self.vocabulary_size2, 1))
        for elem in self.training_data:
            self.items_embeddings[elem[1]] += 1
        self.items_embeddings = np.array(self.items_embeddings)
        print(self.user_embeddings.shape, self.items_embeddings.shape)

        selfplay = Self_Basket_Completion_Model(self)
        calculate_mpr_and_auc_pairs(selfplay, 1, 1, "Disc")


def get_class(model):
    if ("AIS" in model) or (model=="MALIGAN"):
        return Basket_Completion
    else:
        return Self_Basket_Completion

def get_min_steps(dataset):
    return 2500 if (dataset=="Amazon") else 30000

def test_other_models(list_of_models, list_of_datasets, list_of_NS, list_of_networks, task_mode, sampling_type="stochastic"):
    for net in list_of_networks:
        for model in list_of_models:
            for dataset in list_of_datasets:
                for NS in list_of_NS:
                    get_class(model)(dataset=dataset, task_mode=task_mode, model_type=model, min_steps=get_min_steps(dataset), sampling_type=sampling_type, \
                        neg_sampled=NS, G_type=net[0], D_type=net[1])

def switch_launch(argument, neg_sampled):
    if int(argument) == 1:
        test_other_models(["AIS"], ["UK", "Belgian"], [int(neg_sampled)], [("w2v", "w2v")], "user-item")
    if int(argument) == 2:
        test_other_models(["AIS"], ["netflix"], [int(neg_sampled)], [("w2v", "w2v")], "user-item")
    if int(argument) == 3:
        test_other_models(["uniform"], ["UK", "Belgian", "movielens", "netflix", "Amazon"], [int(neg_sampled)], [("w2v", "w2v")], "user-item")
    if int(argument) == 4:
        test_other_models(["baseline"], ["UK"], [int(neg_sampled)], [("w2v", "w2v")], "user-item")
    if int(argument) == 5: 
        test_other_models(["selfplay"], ["netflix"], [int(neg_sampled)], [("w2v", "w2v")], "user-item")
    if int(argument) == 6:
        test_other_models(["softmax"], ["UK", "Belgian", "movielens", "netflix", "Amazon"], [int(neg_sampled)], [("w2v", "w2v")], "user-item")
    if int(argument) == 7:
        test_other_models(["MLE"], ["UK", "Belgian", "movielens", "netflix", "Amazon"], [int(neg_sampled)], [("w2v", "w2v")], "user-item")
    
    if int(argument) == 20:
        test_other_models(["baseline"], ["netflix"], [int(neg_sampled)], [("w2v", "w2v")], "item-item")
    if int(argument) == 21:
        test_other_models(["softmax"], ["text8"], [int(neg_sampled)], [("w2v", "w2v")], "item-item")
    if int(argument) == 22:
        test_other_models(["AIS"], ["text8"], [int(neg_sampled)], [("w2v", "w2v")], "item-item")
    
    if int(argument) == 30:
        #Pre_Loaded_Embedding(dataset="UK", task_mode="item-item", model_type="selfplay")
        #Pre_Loaded_Embedding(dataset="Belgian", task_mode="item-item", model_type="selfplay")
        Pre_Loaded_Embedding(dataset="movielens", task_mode="item-item", model_type="selfplay")
        Pre_Loaded_Embedding(dataset="netflix", task_mode="item-item", model_type="selfplay")
        Pre_Loaded_Embedding(dataset="text8", task_mode="item-item", model_type="selfplay")
       
def usage_several_cpus():
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        executor.map(switch_launch, [1, 2])

switch_launch(sys.argv[1], sys.argv[2])