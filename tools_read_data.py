import copy
from random import shuffle, sample
import itertools
import csv
from tqdm import tqdm
import numpy as np
from collections import Counter
import random
import collections
import math
import os
import sys
import argparse
from tempfile import gettempdir
import zipfile
from word2vec_google import read_data_text8, build_dataset

##############
#Read the different datasets
##############
def print_info_on_data(data, size_threshold):
    Counterr = Counter([len(elem) for elem in data])
    num_seq = len(data)
    num_seq_smaller_than = sum([Counterr[i] for i in range(1,size_threshold+1)])
    print("Number of users = "+ str(len(data)))
    print('max length basket = ', max([len(elem) for elem in data]))
    print('vocabulary size = ', 1+max([max(elem) for elem in data]))
    print('proportion of baskets with size below = ', str(size_threshold),' ', 100*num_seq_smaller_than/num_seq)

def read_data(data_file_path):
    data = [[]]
    with open(data_file_path) as data_csv_file:
        data_csv_reader = csv.reader(data_csv_file, delimiter = ',')
        for row in data_csv_reader:
            if len(row) == 1:
                continue
            data.append([(int(i) - 1) for i in row])
    del data[0]
    return data

def get_data(input_file):
    data = list()
    with open(input_file) as f:
        for line in f:
            sub_data = line.split(" ")
            data.append([int(elem) for elem in sub_data if elem != "\n"])
    return data

def get_data_movies_datasets(input_file):
    data = list()
    with open(input_file) as f:
        for line in f:
            sub_data = line.split(",")
            if ("netflix" in input_file):
                data.append([int(float(elem)) for elem in sub_data if elem != "\n"])
            else:
                data.append([int(elem) for elem in sub_data if elem != "\n"])
    return data

def read_data_UK_retail(data_file_path):
    dict_basket = dict()
    dict_from_items_to_ids = dict()
    dict_from_items_to_desc = dict()
    with open(data_file_path, encoding='utf-8') as data_csv_file:
        data_csv_reader = csv.reader(data_csv_file, delimiter = ',')
        i, num_of_ids = 0, 0
        for row in data_csv_reader:
            if (i>0):
                if (row[0] in dict_basket):
                    if (row[1] in dict_from_items_to_ids):
                        dict_basket[row[0]].append(dict_from_items_to_ids[row[1]])
                    else:
                        dict_from_items_to_ids[row[1]] = num_of_ids
                        dict_from_items_to_desc[row[1]] = row[2]
                        dict_basket[row[0]].append(dict_from_items_to_ids[row[1]])
                        num_of_ids += 1
                else:
                    if (row[1] in dict_from_items_to_ids):
                        dict_basket[row[0]] = [dict_from_items_to_ids[row[1]]]
                    else:
                        dict_from_items_to_ids[row[1]] = num_of_ids
                        dict_from_items_to_desc[row[1]] = row[2]
                        dict_basket[row[0]] = [dict_from_items_to_ids[row[1]]]
                        num_of_ids += 1
            i += 1
    baskets = [dict_basket[key] for key in sorted(list(dict_basket.keys()))]
    dict_from_ids_to_descriptions = {v: dict_from_items_to_desc[k] for k, v in dict_from_items_to_ids.items()}
    print(num_of_ids)
    return baskets, dict_from_ids_to_descriptions

def load_NYtimes_dataset(data_file_path):
    list_of_baskets = list([])
    dict_word_to_ids = dict()
    i = 0
    with open(data_file_path) as data_csv_file:
        data_csv_reader = csv.reader(data_csv_file, delimiter = ' ')
        for row in data_csv_reader:
            word = row[0].strip("\t")
            word = word.strip("zzz")
            if (word=='Topic'):
                list_of_baskets.append([])
            else:
                if (word in dict_word_to_ids):
                    list_of_baskets[-1].append(dict_word_to_ids[word])
                else:
                    dict_word_to_ids[word] = i
                    i += 1
                    list_of_baskets[-1].append(dict_word_to_ids[word])
    return np.array(list_of_baskets)

def create_text8_dataset(filename):
    data = read_data_text8(filename)
    n_words, top_words_removed_threshold = 12000, 25
    data, count, dictionary, reversed_dictionary = \
        build_dataset(data, n_words, top_words_removed_threshold)
    pairs = np.array([[data[i], data[i+1]] for i in range(len(data)-1)])
    return pairs, reversed_dictionary
    

def load_data(self):
    dictionnary = {}
    if self.dataset == "text8":
        data, dictionnary = create_text8_dataset("datasets/text8.zip")
        folder = "datasets/text8/"
        metric = "MPR"
        type_of_data = "real"
    if self.dataset == "NYtimes":
        data = load_NYtimes_dataset("datasets/NYtimes.twords")
        folder = "datasets/NYtimes/"
        metric = "MPR"
        type_of_data = "real"
    if self.dataset == "movielens":
        data = get_data_movies_datasets("datasets/movielens_new.csv")
        folder = "datasets/movielens/"
        metric = "MPR"
        type_of_data = "real"
    if self.dataset == "netflix":
        data = get_data_movies_datasets("datasets/netflix_new.csv")
        folder = "datasets/netflix/"
        metric = "MPR"
        type_of_data = "real"
    if self.dataset == "netflix_mf_Ponly":
        data = np.load("datasets/netflix_mf_Ponly.npy")
        folder = "datasets/netflix/"
        metric = "AUC"
        type_of_data = "real"
    if self.dataset == "movielens_mf_Ponly":
        data = np.load("datasets/movielens_mf_Ponly.npy")
        folder = "datasets/netflix/"
        metric = "AUC"
        type_of_data = "real"
    if self.dataset == "Belgian":
        data = get_data("datasets/retail.dat")
        folder = "datasets/Belgian/"
        metric = "MPR"
        type_of_data = "real"
    if self.dataset == "UK":
        data, dictionnary = read_data_UK_retail("datasets/UK_retail2.csv")
        folder = "datasets/UK/"
        metric = "MPR"
        type_of_data = "real"
    if self.dataset == "Amazon":
        data = read_data("datasets/1_100_100_100_apparel_regs.csv")
        folder = "datasets/Amazon/"
        metric = "MPR"
        type_of_data = "real"
    if self.dataset == "swiss_roll":
        data = np.load("manifolds/datasets/swiss_roll/true_data.npy")
        folder = "manifolds/swiss_roll/"
        metric = "AUC"
        type_of_data = "synthetic"
    if self.dataset == "s_curve":
        data = np.load("manifolds/datasets/s_curve/true_data.npy")
        folder = "manifolds/s_curve/"
        metric = "AUC"
        type_of_data = "synthetic"
    if self.dataset == "blobs":
        data = np.load("manifolds/datasets/blobs/true_data.npy")
        folder = "manifolds/blobs/"
        metric = "AUC"
        type_of_data = "synthetic"
    if self.dataset == "moons":
        data = np.load("manifolds/datasets/moons/true_data.npy")
        folder = "manifolds/moons/"
        metric = "AUC"
        type_of_data = "synthetic"
    if self.dataset == "blobs0":
        data = np.load("manifolds/datasets/blobs0/true_data.npy")
        folder = "manifolds/blobs0/"
        metric = "AUC"
        type_of_data = "synthetic"
    if self.dataset == "blobs1":
        data = np.load("manifolds/datasets/blobs1/true_data.npy")
        folder = "manifolds/blobs1/"
        metric = "AUC"
        type_of_data = "synthetic"
    if self.dataset == "blobs2":
        data = np.load("manifolds/datasets/blobs2/true_data.npy")
        folder = "manifolds/blobs2/"
        metric = "AUC"
        type_of_data = "synthetic"
    if self.dataset == "circles":
        data = np.load("manifolds/datasets/circles/true_data.npy")
        folder = "manifolds/circles/"
        metric = "AUC"
        type_of_data = "synthetic"
    
    print_info_on_data(data, self.max_basket_size)
    data = [sample(elem, min(len(elem), self.max_basket_size)) for elem in data]
    return data, folder, metric, type_of_data, dictionnary