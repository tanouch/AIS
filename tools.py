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

def create_dir(type_of_data):
    if not os.path.exists(type_of_data):
        os.makedirs(type_of_data)

def get_co_occurences_dict(data, task_mode):
    dic = {}
    for i, elem in enumerate(data):
        if (task_mode == "item-item"):
            for e in elem:
                if e in dic:
                    dic[e].update(elem)
                else:
                    dic[e] = Counter(elem)
        elif (task_mode == "user-item"):
            dic[i] = Counter(elem)
    return dic

def list_elem_not_co_occuring(self, item):
    if item not in self.model_params.dict_co_occurences:
        return np.append(np.arange(self.model_params.vocabulary_size2), item)
    else:
        return np.append(np.delete(np.arange(self.model_params.vocabulary_size2), list(self.model_params.dict_co_occurences[item])), item)

def get_negatives(self, item, size):
    co_occuring = list(self.model_params.dict_co_occurences[item])
    return np.random.choice(np.delete(np.arange(self.model_params.vocabulary_size2), co_occuring), size)


def get_the_density_of_the_baskets_length_in_the_data(data):
    length = [len(elem) for elem in data]
    total = len(length)
    counter = Counter(length)
    density = [counter[elem]*100/total for elem in range(25)]
    cumul = np.column_stack((np.arange(25),np.cumsum(density)))
    print(cumul)

def create_item_to_item_data(dataset, listoflist, rnd):
    all_pairs = list()
    for elem in listoflist:
        if len(elem)<35:
            all_pairs.extend(list(itertools.combinations(elem, 2))[:50])
        else:
            all_pairs.extend(list(itertools.combinations(sample(elem, 35), 2))[:50])
    arr = np.array(all_pairs)
    rnd.shuffle(arr)
    print("Number of training pairs", len(arr))
    return arr

def create_user_to_item_data(dataset, listoflist, rnd):
    if (dataset=="UK") or (dataset=="Belgian") or (dataset=="netflix") or (dataset=="movielens"):
        all_pairs = list()
        for i, elem in enumerate(listoflist):
            for e in elem:
                all_pairs.append([i, e])
        arr = np.array(all_pairs)
        rnd.shuffle(arr)
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
    if (self.type_of_data=="synthetic"):
        self.X, self.Y, self.Z, self.threshold = np.load(self.folder+"X.npy"), np.load(self.folder+"Y.npy"), np.load(self.folder+"Z.npy"), np.load(self.folder+"threshold.npy")
    if (self.type_of_data=="real") and (self.dataset != "text8"):
        shuffle(self.data)
    if (self.type_of_data=="synthetic") or (self.dataset=="text8"):
        np.random.shuffle(self.data)

def get_vocabulary_size(self):
    if (self.task_mode=="item-item"):    
        if (self.type_of_data=="synthetic"):
            self.vocabulary_size = 10000
        else:
            self.vocabulary_size = 1+max([elem for ss in self.data for elem in ss])      
        self.vocabulary_size2 = self.vocabulary_size
    elif (self.task_mode=="user-item"):
        self.vocabulary_size, self.vocabulary_size2 = (1+max([elem for elem in self.data[:,0]]), 1+max([elem for elem in self.data[:,1]]))
    return (self.vocabulary_size, self.vocabulary_size2)

def split_data(data):
    proportion_training, proportion_test = (0.8, 0.2)
    num_test_instances = int(len(data)*proportion_test)
    test_data, training_data = data[:num_test_instances], data[num_test_instances:]
    return test_data, training_data

def he_xavier(in_size, out_size, names):
    stddev = tf.cast(tf.sqrt(2 / in_size), tf.float32)
    initializer = tf.contrib.layers.xavier_initializer(uniform=False)
    W = tf.get_variable(name= names[0], shape=[in_size, out_size], initializer=initializer)
    b = tf.get_variable(name= names[1], shape=[1,1], initializer=tf.constant_initializer(0.0, dtype=tf.float32))
    return W, b

def generate_wholeBatch(data, skip_window, generate_all_pairs):
        context = list()
        labels = list()
        for i, sequence in enumerate(data):
            length = len(sequence)
            for j in range(length):
                sub_list = list()
                if (j - skip_window) >= 0 and (j + skip_window) < length:
                    sub_list += sequence[j - skip_window:j]
                    sub_list += sequence[j + 1:j + skip_window + 1]
                elif j > skip_window:  # => j+skip_window >= length
                    sub_list += sequence[j - skip_window:j]
                    if j < (length - 1):
                        sub_list += sequence[j + 1:length]
                elif (j + skip_window) < length:
                    if j > 0:
                        sub_list += sequence[0:j]
                    sub_list += sequence[j + 1:j + skip_window + 1]
                else:
                    if j > 0:
                        sub_list += sequence[0:j]
                    if j < length - 1:
                        sub_list += sequence[j + 1:length]
                    if length == 1:
                        sub_list += sequence[j:j + 1]

                if (generate_all_pairs==True):
                    for elem in sub_list:
                        context.append(elem)
                        labels.append(sequence[j])
                else:
                    context.append(sub_list)
                    labels.append(sequence[j])

        labels = np.array(labels)
        array_length = len(labels)
        labels.shape = (array_length,1)
        context = np.array(context)
        shuffle_idx = np.random.permutation(array_length)
        labels = labels[shuffle_idx]
        context = context[shuffle_idx]
        return context, labels