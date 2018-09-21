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
import tensorflow_glove.tf_glove as tf_glove

from tools import *
from tools_read_data import *
from gen_cnn import Gen_CNN_Model
from word2vec import W2V_Model
from metric_utils import print_results_predictions, calculate_mpr_and_auc_pairs, check_similarity, check_analogies
from self_basket_completion import Self_Basket_Completion_Model
from basket_completion import Basket_Completion_Model

class Model(object):
    def __init__(self, dataset, task_mode, sampling):
        self.seed = 1234
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)
        self.rnd = np.random.RandomState(1234)
        
        self.embedding_size, self.batch_size, self.epoch, self.speeding_factor, self.seq_length = 150, 200, 10, 25, 2
        self.conv_filters = [2, 4, 6, 8, 12]
        self.use_pretrained_embeddings=False

        #Get data
        self.max_basket_size, self.test_data_max_size = 50, 5000
        self.dataset, self.task_mode, self.working_with_pairs = dataset, task_mode, True
        self.data, self.folder, self.metric, self.type_of_data, self.dictionnary = load_data(self)
        
        if task_mode=="item-item":
            self.data = create_item_to_item_data(dataset, self.data, self.rnd)
        if task_mode=="user-item":
            self.data = create_user_to_item_data(dataset, self.data, self.rnd)

        self.dict_co_occurences = get_co_occurences_dict(self)
        self.vocabulary_size, self.vocabulary_size2 = get_vocabulary_size(self)
        if (sampling=="context"):
            self.conditional_distributions = get_context_conditional_distributions(self)
        load_threshold_and_Z(self)
        self.test_data, self.training_data = split_data(self.data, 0.8)
        #get_mutual_exclusivity_stats(self)
        
        self.check_words_similarity = np.array(list(self.rnd.choice(100, 25, replace=False)) \
            + list(500 + self.rnd.choice(100, 25, replace=False))
            + list(1000 + self.rnd.choice(100, 25, replace=False))
            + list(3000 + self.rnd.choice(100, 25, replace=False)))
        if (dataset=="text8"):
            self.list_semantic_analogies, self.list_syntactic_analogies = np.load("datasets/semantics12000.npy"), np.load("datasets/syntactics12000.npy")
        else:
            self.list_semantic_analogies, self.list_syntactic_analogies = np.load("datasets/semantics30000.npy"), np.load("datasets/syntactics30000.npy")

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

def create_all_attributes(self, dataset, model_type, neg_sampled, G_type, D_type, sampling):
        self.use_pretrained_embeddings, self.embedding_matrix = False, 0
        self.G_type, self.D_type = G_type, D_type
        self.model_type, self.neg_sampled = model_type, neg_sampled
        self.min_steps = 15000
        self.max_steps = 30000 if (dataset in ["blobs0", "swiss_roll", "s_curve", "moons"]) else 75000
        self.printing_step, self.saving_step = 5000, 2500
        self.name = model_type + "_" + dataset +"_"+ G_type + "_" + str(neg_sampled)
        if (sampling=="context") or (sampling=="uniform") or (sampling=="selfplay"):
            self.name = self.name + "_" + sampling
        self.adv_generator_loss = ["AIS", "Not_Mixed"]
        self.negG, self.negD = [1,0], [1,0]
        self.discriminator_samples_type = sampling
        
        if (model_type=="AIS"):
            self.adv_generator_loss, self.adv_discriminator_loss = ["AIS", "Not_Mixed"], ["SS", "Not_Mixed"]
            self.negD = [15, 0] if (neg_sampled<10) else [neg_sampled, 0]
            self.negG = [5, 0] if (neg_sampled<1) else [neg_sampled, 0]

        elif (model_type=="AIS-BCE"):
            self.adv_generator_loss, self.adv_discriminator_loss = ["AIS-BCE", "Not_Mixed"], ["BCE", "Not_Mixed"]
            self.negD = [neg_sampled, 0]
            self.negG = [5, 0] if (neg_sampled<1) else [neg_sampled, 0]
        
        elif (model_type=="MALIGAN"):
            self.adv_generator_loss, self.adv_discriminator_loss = ["MALIGAN", "Mixed"], ["SS", "Not_Mixed"]
            self.negD = [neg_sampled, 0]
            self.negG = [5, 0] if (neg_sampled<1) else [neg_sampled, 0]
        
        elif (model_type=="softmax") or (model_type=="MLE"):
            self.adv_discriminator_loss = [model_type, "Not_Mixed"]
        
        elif (model_type=="SS"):
            self.adv_discriminator_loss = ["SS", "Not_Mixed"]
        
        elif (model_type=="BCE"):
            self.adv_discriminator_loss = ["BCE", "Not_Mixed"]

        elif (model_type=="baseline"):
            self.adv_discriminator_loss = ["baseline", "Not_Mixed"]
        
        if (model_type=="SS") or (model_type=="BCE"):
            self.negD = [neg_sampled, 0] if (sampling=="selfplay") else [0, neg_sampled]
        self.generator_samples_type = "selfplay"
        print(dataset, model_type, neg_sampled, "negD", self.negD, "negG", self.negG, "G_type", G_type, "D_type", D_type, "sampling", sampling)

class Basket_Completion(Model):
    def __init__(self, dataset, task_mode, model_type, neg_sampled, G_type, D_type, sampling):
        Model.__init__(self, dataset, task_mode, sampling)
        create_all_attributes(self, dataset, model_type, neg_sampled, G_type, D_type, sampling)
        gan = Basket_Completion_Model(self)
        gan.train_model_with_tensorflow()
        del gan

class Self_Basket_Completion(Model):
    def __init__(self, dataset, task_mode, model_type, neg_sampled, G_type, D_type, sampling):
        Model.__init__(self, dataset, task_mode, sampling)
        create_all_attributes(self, dataset, model_type, neg_sampled, G_type, D_type, sampling)
        self_play = Self_Basket_Completion_Model(self)
        self_play.train_model_with_tensorflow()
        del self_play

class Check_Bestof_Method(Model):
    def __init__(self, dataset, task_mode, model_type, neg_sampled, G_type, D_type, sampling):
        Model.__init__(self, dataset, task_mode, sampling)
        create_all_attributes(self, dataset, model_type, neg_sampled, G_type, D_type, sampling)
        self.use_pretrained_embeddings = True        
        self.user_embeddings = np.ones((self.vocabulary_size,1))
        self.item_embeddings = np.zeros((self.vocabulary_size2, 1))
        for elem in self.training_data:
            self.item_embeddings[elem[1]] += 1
        self.item_embeddings = np.array(self.item_embeddings)
        selfplay = Self_Basket_Completion_Model(self)
        calculate_mpr_and_auc_pairs(selfplay, 1, [[], [], [], [], [], []], "Disc")

class Check_Embedings(Model):
    def __init__(self, dataset, task_mode, model_type, neg_sampled, G_type, D_type, sampling, list_of_emb_files=[]):
        Model.__init__(self, dataset, task_mode, sampling)
        create_all_attributes(self, dataset, model_type, neg_sampled, G_type, D_type, sampling)
        self.use_pretrained_embeddings = True
        
        for emb_name in list_of_emb_files:
            emb_name = emb_name
            self.name = emb_name
            self.item_embeddings = np.load(emb_name)
            selfplay = Self_Basket_Completion_Model(self)
            check_similarity(selfplay, 0)
            check_analogies(selfplay, 0)

class Glove(Model):
    def __init__(self, dataset, task_mode, model_type, neg_sampled, G_type, D_type, sampling):
        self.dataset = dataset
        self.rnd, self.embedding_size, self.batch_size, self.use_pretrained_embeddings, self.name = np.random.RandomState(1234), 150, 200, True, "glove"
        self.check_words_similarity = np.array(list(self.rnd.choice(100, 25, replace=False)) \
            + list(500 + self.rnd.choice(100, 25, replace=False))
            + list(1000 + self.rnd.choice(100, 25, replace=False))
            + list(3000 + self.rnd.choice(100, 25, replace=False)))
        if (dataset=="text8"):
            self.list_semantic_analogies, self.list_syntactic_analogies = np.load("datasets/semantics12000.npy"), np.load("datasets/syntactics12000.npy")
        else:
            self.list_semantic_analogies, self.list_syntactic_analogies = np.load("datasets/semantics30000.npy"), np.load("datasets/syntactics30000.npy")
        
        data = read_data_text8("datasets/text8.zip")
        n_words, top_words_removed_threshold = 30000, 25
        data, count, dictionary, reversed_dictionary = \
            build_dataset(data, n_words, top_words_removed_threshold)
        
        glove = tf_glove.GloVeModel(embedding_size=self.embedding_size, max_vocab_size=12001,context_size=1, min_occurrences=0,
                            learning_rate=0.05, batch_size=self.batch_size)
        glove.fit_to_corpus([data])
        for i in range(20):
            glove.train(num_epochs=5)
            np.save("glove_emb"+str(5+i*5),glove.embeddings)

def get_class(model):
    if ("AIS" in model) or (model=="MALIGAN"):
        return Basket_Completion
    if (model=="SS") or (model=="BCE") or (model=="softmax") or (model=="MLE") or (model=="baseline"):
        return Self_Basket_Completion
    if (model=="glove"):
        return Glove

def test_other_models(list_of_models, list_of_datasets, list_of_NS=[1], list_of_networks=["w2v", "w2v"], sampling="uniform", task_mode="item-item"):
    for net in list_of_networks:
        for model in list_of_models:
            for dataset in list_of_datasets:
                for NS in list_of_NS:
                    get_class(model)(dataset=dataset, task_mode=task_mode, model_type=model, \
                        neg_sampled=NS, G_type=net[0], D_type=net[1], sampling=sampling)

def switch_launch(argument, neg_sampled):
    if int(argument) == 1:
        test_other_models(["AIS"], ["blobs0", "swiss_roll", "s_curve", "moons", "UK", "Belgian", "text8"], [int(neg_sampled)], [("w2v", "w2v")], "selfplay")
    if int(argument) == 2:
        test_other_models(["SS"], ["blobs0", "swiss_roll", "s_curve", "moons", "UK", "Belgian", "text8"], [int(neg_sampled)], [("w2v", "w2v")], "selfplay")
    if int(argument) == 3:
        test_other_models(["baseline"], ["blobs0", "swiss_roll", "s_curve", "moons", "UK", "Belgian", "text8"], [int(neg_sampled)], [("w2v", "w2v")])
    if int(argument) == 4:
        test_other_models(["SS"], ["blobs0", "swiss_roll", "s_curve", "moons", "UK", "Belgian", "text8"], [int(neg_sampled)], [("w2v", "w2v")], "uniform",)
    if int(argument) == 5:
        test_other_models(["softmax"], ["blobs0", "swiss_roll", "s_curve", "moons", "UK", "Belgian", "text8"], [int(neg_sampled)], [("w2v", "w2v")])
    if int(argument) == 6:
        test_other_models(["MLE"], ["blobs0", "swiss_roll", "s_curve", "moons", "UK", "Belgian"], [int(neg_sampled)], [("w2v", "w2v")])
    if int(argument) == 7:
        test_other_models(["SS"], ["blobs0", "swiss_roll", "s_curve", "moons", "UK", "Belgian"], [int(neg_sampled)], [("w2v", "w2v")], "context")

    if int(argument) == 20:
        test_other_models(["baseline"], ["text8"], [int(neg_sampled)], [("w2v", "w2v")], "item-item")
    if int(argument) == 21:
        test_other_models(["softmax"], ["text8"], [int(neg_sampled)], [("w2v", "w2v")], "item-item")
    if int(argument) == 22:
        test_other_models(["selfplay"], ["text8"], [int(neg_sampled)], [("w2v", "w2v")], "item-item")
    if int(argument) == 23:
        test_other_models(["AIS"], ["text9"], [int(neg_sampled)], [("w2v", "w2v")], "item-item")
    if int(argument) == 24:
        test_other_models(["AIS"], ["text8"], [int(neg_sampled)], [("w2v", "w2v")], "item-item")
    if int(argument) == 25:
        test_other_models(["AIS-BCE"], ["text9"], [int(neg_sampled)], [("w2v", "w2v")], "item-item")

    if int(argument) == 30:
        Check_Embedings(dataset="UK", task_mode="item-item", model_type="glove", list_of_emb_files= \
        ["AIS_text9_w2v_6_gen_emb32500.npy"])
       
def usage_several_cpus():
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        executor.map(switch_launch, [1, 2])

switch_launch(sys.argv[1], sys.argv[2])
#switch_launch(30, 1)
#read_text_file("text8_30000_25.vec", "fastText_25.npy")
