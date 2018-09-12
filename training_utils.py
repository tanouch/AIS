import pickle
from datetime import datetime
import argparse
import os
import tensorflow as tf
import math
import numpy as np
import time
from word2vec_google import *
from tools import *


def check_step_equal_end_of_pretraining(self, step, number_of_pretraining_steps):
    if (step==number_of_pretraining_steps):
        print("######################")
        print("Adversarial training")
        print("######################")

def get_pairs_batch(index, size, data):
    if (index+size < len(data)):
            batches = data[index:index+size]
            index += size
    else:
        batches = data[index:]
        index = 0
    return index, batches

def create_batch(self, size):
    if (self.model_params.metric == "AUC"):
        self.index, batches = get_pairs_batch(self.index, size, self.training_data)
    
    else :
        if (self.model_params.working_with_pairs):
            self.index, batches = get_pairs_batch(self.index, size, self.training_data)
        else:
            seq_length = np.random.randint(8)+2
            batches = get_basket_set_size_training(self.training_data, size, seq_length)
    
    train_words, label_words = batches[:,:-1], batches[:,1:]
    return train_words, label_words

def training_step(self, list_of_operations_to_run, batch_size):
    train_words, label_words = create_batch(self, batch_size)
    return self._sess.run(list_of_operations_to_run,
        feed_dict={self.train_words:train_words, self.label_words:label_words, self.dropout:1})

def print_gen_and_disc_losses(self, step):
    print("")
    print("Step "+ str(step))
    if (step==0):
        print("Gen losses " + str(self.Gen_loss1) + "  "+ str(self.Gen_loss2))
        print("Disc losses " + str(self.Disc_loss1) + "  "+ str(self.Disc_loss2))
    else:
        print("Gen losses " + str(self.Gen_loss1/self.model_params.printing_step) + "  "+ str(self.Gen_loss2/self.model_params.printing_step))
        print("Disc losses " + str(self.Disc_loss1/self.model_params.printing_step) + "  "+ str(self.Disc_loss2/self.model_params.printing_step))
    print("")
    self.Gen_loss1, self.Gen_loss2, self.Disc_loss1, self.Disc_loss2 = 0, 0, 0, 0

def early_stopping(main_score, threshold):
    if (main_score[-1] < np.mean(main_score[-threshold:-2])) and (main_score[-2] < np.mean(main_score[-(threshold+1):-3])):
        return True
    else:
        return False
