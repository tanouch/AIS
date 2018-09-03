import copy
from random import shuffle, sample
import itertools
import csv
from tqdm import tqdm
import numpy as np
from collections import Counter
import random
from word2vec_google import *
from tools_read_data import print_info_on_data

def get_the_density_of_the_baskets_length_in_the_data(data):
    length = [len(elem) for elem in data]
    total = len(length)
    counter = Counter(length)
    density = [counter[elem]*100/total for elem in range(25)]
    cumul = np.column_stack((np.arange(25),np.cumsum(density)))
    print(cumul)

def create_2D_datasets_from_list_of_list(listoflist):
    all_pairs = list()
    for elem in listoflist:
        if len(elem)<35:
            all_pairs.extend(list(itertools.combinations(elem, 3))[:50])
        else:
            all_pairs.extend(list(itertools.combinations(sample(elem, 35), 3))[:50])
    arr = np.array(all_pairs)
    np.random.shuffle(arr)
    print("Number of training pairs", len(arr))
    return arr

def counters_per_prod(data):
    counter = Counter()
    for elem in data:
        counter.update(elem)
    list_of_keys = np.unique(np.array(list(counter.keys())))
    print("Vocabulary size with unique items", len(list_of_keys))
    return counter, len(list_of_keys)

def get_basket_set_size(data, seq_length):
    batch = list()
    cont = True
    for basket in data:
        if len(basket)==seq_length:
            batch.append(basket)
    return np.array(batch)

def get_basket_set_size_training(data, batch_size, seq_length):
    batch = list()
    while len(batch)<batch_size:
        line = np.random.randint(0, len(data))
        basket = data[line]
        if len(basket)>=seq_length:
            batch.append(list(np.random.choice(basket, size=seq_length, replace=False)))
    return np.array(batch)

def get_test_list_batches(self):
    test_data = copy.copy(self.test_data)[:self.test_data_max_size]
    self.test_list_batches = list()
    if (self.type_of_data=="real"):
        for elem in test_data:
            random.shuffle(elem)
    for i in range(2, self.max_basket_size):
        if (self.type_of_data=="real") and (self.dataset!="NYtimes") and (self.dataset!="text8"):
            self.test_list_batches.append(get_basket_set_size(test_data, i))
    self.test_list_batches.insert(0, [])
    self.test_list_batches.insert(0, [])

    self.test_baskets = [self.test_data[i] for i in sorted(random.sample(range(len(self.test_data)), min(2500, len(self.test_data))))]

    self.test_items = np.reshape(np.array([i for i in range(25, 50)] + [i for i in range(100, 125)] + [i for i in range(250, 275)] \
                + [i for i in range(500, 525)] + [i for i in range(1000, 1025)]), (-1,1))
    return self.test_list_batches, self.test_baskets, self.test_items

def get_popularity_dist(training_data, vocabulary_size):
    counter = Counter()
    if (type(training_data[0])==int):
        counter.update(training_data)
    else:
        for elem in training_data:
            counter.update(elem)
    popularity_dist = [float(counter[i]) for i in range(vocabulary_size)]
    return popularity_dist

def write_results_in_csv(filename, list_of_list):
    with open(filename,"w") as f :
        csv_writer = csv.writer(f)
        csv_writer.writerow(zip(list_of_list[0], list_of_list[1], list_of_list[2], list_of_list[3], list_of_list[4], list_of_list[5]))

def process_and_shuffle_data(self):
    if (self.dataset=="text8"):
        top_words_removed_threshold = 25
        self.data, self.count, self.dictionary, self.reverse_dictionary = build_dataset(self.data, self.vocabulary_size, top_words_removed_threshold)
        self.data_index = 0
    if (self.type_of_data=="synthetic"):
        self.X, self.Y, self.Z, self.threshold = np.load(self.folder+"X.npy"), np.load(self.folder+"Y.npy"), np.load(self.folder+"Z.npy"), np.load(self.folder+"threshold.npy")
    if (self.metric=="MPR"):
        shuffle(self.data)
    if (self.metric=="AUC"):
        np.random.shuffle(self.data)

def get_vocabulary_size(self):
    if (self.type_of_data=="synthetic"):
        self.vocabulary_size, self.vocabulary_size2 = 10000, 10000
    elif (self.dataset=="text8"):
        self.vocabulary_size, self.vocabulary_size2 = 20000, 20000
    elif ("mf" in self.dataset):
        self.vocabulary_size, self.vocabulary_size2 = (1+max([elem for elem in self.data[:,0]]), 1+max([elem for elem in self.data[:,1]]))
    else:
        self.vocabulary_size = 1+max([elem for ss in self.data for elem in ss])      
        self.vocabulary_size2 = self.vocabulary_size
    return (self.vocabulary_size, self.vocabulary_size2)

