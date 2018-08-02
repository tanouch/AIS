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

def create_batch(self):
    seq_length = np.random.randint(8)+2
    if (self.model_params.type_of_data == "synthetic") or (self.model_params.dataset == "NYtimes"):
        batches = self.training_data[np.random.choice(len(self.training_data), size=self.batch_size, replace=False)]
    elif (self.model_params.dataset == "text8"):
        batches = generate_batch_google(self, num_skips=2, skip_window=1)
    else : #(self.model_params.type_of_data == "real"):
        batches = get_basket_set_size_training(self.training_data, self.batch_size, seq_length)
    train_words, label_words = batches[:,:-1], batches[:,1:]
    return train_words, label_words

def training_step(self, list_of_operations_to_run):
    train_words, label_words = create_batch(self)
    return self._sess.run(list_of_operations_to_run,
        feed_dict={self.train_words:train_words, self.label_words:label_words, self.dropout:1})

def print_gen_and_disc_losses(self, step):
    print("Step "+ str(step))
    print("Gen losses " + str(self.Gen_loss1/self.printing_step) + "  "+ str(self.Gen_loss2/self.printing_step))
    print("Disc losses " + str(self.Disc_loss1/self.printing_step) + "  "+ str(self.Disc_loss2/self.printing_step))
    print("")
    self.Gen_loss1, self.Gen_loss2, self.Disc_loss1, self.Disc_loss2 = 0, 0, 0, 0
