import copy
import math
import os
import csv
import random
from tqdm import tqdm
from keras.layers.embeddings import Embedding
import numpy as np
import tensorflow as tf
from metric_utils import *
from tools import *

class W2V_Model(object):

    def __init__(self, wider_model):
        #w2v.__init__(self)
        self.vocabulary_size, self.vocabulary_size2 = wider_model.vocabulary_size, wider_model.vocabulary_size2
        self.embedding_size = wider_model.embedding_size
        self.batch_size = wider_model.batch_size
        self.num_sampled = wider_model.neg_sampled_pretraining
        self.num_epochs = wider_model.epoch

        self.embedding_matrix = wider_model.embedding_matrix
        self.seq_length = wider_model.seq_length
        self.use_pretrained_embeddings = wider_model.use_pretrained_embeddings

        self.model_type = "cbow"
        self.loss = "NEG"
        self.learning_rate = 2
        self.printing_step = 1000

        self.training_data, self.test_data = wider_model.training_data, wider_model.test_data
        #self.X_train, _, self.Y_train = preprocessing_data(wider_model.training_data, wider_model.seq_length, 'cnn')
        #self.X_test, _, self.Y_test = preprocessing_data_one_shuffle(wider_model.test_data, wider_model.seq_length, 'cnn') #no shuffle for test

    def generate_wholeBatch(self, data, skip_window, generate_all_pairs):
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

    def create_embedding_layer(self):
        with tf.name_scope("embeddings"):
            with tf.device('/cpu:0'):
                if self.use_pretrained_embeddings:
                    print("Initializing with the previous embedding matrix")
                    self.embeddings = Embedding(self.vocabulary_size, self.embedding_size, weights=[self.embedding_matrix], input_length=self.seq_length, trainable=True)
                else:
                    print("Initializing with a random embedding matrix")
                    self.embeddings = Embedding(self.vocabulary_size, self.embedding_size, embeddings_initializer='uniform', input_length=self.seq_length)
        return self.embeddings

    def creating_layer(self, input_tensor, dropout=None):
        self.embed = self.embeddings(input_tensor) if (self.model_type=="skip_gram") else \
            tf.reduce_mean(self.embeddings(input_tensor), axis=1)
        return self.embed

    def compute_loss(self, output, label_tensor):
        with tf.name_scope("output_layer"):
            self.nce_weights = tf.Variable(tf.truncated_normal([self.vocabulary_size2, self.embedding_size], stddev=1.0 / math.sqrt(self.embedding_size)))
            self.nce_biases = tf.Variable(tf.zeros([self.vocabulary_size2]))

            if (self.loss == "NCE"):
                self.loss = tf.reduce_mean(tf.nn.nce_loss(
                    weights=self.nce_weights, biases=self.nce_biases, labels=label_tensor,
                    inputs=output, num_sampled=self.num_sampled, num_classes=self.vocabulary_size2))
            if (self.loss == "NEG"):
                self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
                    weights=self.nce_weights, biases=self.nce_biases, labels=label_tensor,
                    inputs=output, num_sampled=self.num_sampled, num_classes=self.vocabulary_size2))
        return self.loss

    def get_predictions(self, output):
        self.last_predictions = tf.nn.softmax(tf.matmul(output,tf.transpose(self.nce_weights))+self.nce_biases)
        return self.last_predictions
    def get_all_scores(self, output):
        self.before_softmax = tf.matmul(output,tf.transpose(self.nce_weights))+self.nce_biases
        return self.before_softmax
    def get_score(self, output, elem):
        nce_weights, nce_biases = tf.nn.embedding_lookup(self.nce_weights, elem), tf.nn.embedding_lookup(self.nce_biases, elem)
        return tf.reduce_sum(tf.multiply(output, nce_weights), axis=1) + nce_biases
    def get_similarity(self, input_tensor):
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings_tensorflow), 1, keepdims=True))
        normalized_embeddings = self.embeddings_tensorflow / norm
        output = tf.reduce_mean(tf.nn.embedding_lookup(normalized_embeddings, input_tensor), axis=1)
        self.similarity = tf.matmul(output, tf.transpose(normalized_embeddings))
        return self.similarity

    def create_graph(self):
        with tf.name_scope("inputs"):
            self.train_inputs = tf.placeholder(tf.int32, shape=[None])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
            self.context_inputs = tf.placeholder(tf.int32, shape=[None, self.seq_length])

        self.W = self.create_embedding_layer()
        self.output = self.creating_layer(self.train_inputs) if (self.model_type=="skip_gram") else self.creating_layer(self.context_inputs)
        self.compute_loss(self.output, self.train_labels)
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        self.last_predictions = self.get_predictions(self.output)
        self.embeddings_looked_up = self.embeddings(self.train_inputs)

    def train_model_with_tensorflow(self):
        self.create_graph()
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())
        print('Initialized')

        context,labels = self.generate_wholeBatch(self.training_data, skip_window = 6, generate_all_pairs = False)
        data_idx, epoch, step, average_loss = 0, 0, 0, 0

        while (step<15000):
            print1 = 250
            self.printing_step = 250

            if (self.model_type=="skip_gram"):
                batch_inputs, batch_labels, data_idx = generate_batch(self.batch_size, context, labels, data_idx)
                feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}
            else:
                batch_inputs, batch_labels = get_the_next_batch_random(self.X_train, self.Y_train, self.batch_size)
                feed_dict = {self.context_inputs: batch_inputs, self.train_labels: np.reshape(batch_labels,(-1,1))}

            _, loss_val = self._sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
            step += 1
            average_loss += loss_val

            if (step % 1000 == 0) and (step>0):
                average_loss /= self.printing_step
                print("Epoch "+ str(epoch) + " step "+ str(step) +" and loss " + str(average_loss))
                average_loss = 0

            if (step % 1000 == 0) and (step>0):
                for i in range(6, 10):
                    batches = get_basket_set_size(self.test_data, 500, i)
                    train_words, targets = batches[:,:-1], np.reshape(batches[:,-1], (-1,1))
                    #MPR with G
                    last_predictions = self._sess.run([self.output_distributions], feed_dict={self.train_words:train_words, self.dropout:1})[0]
                    print("########"+ str(i)+ "########")
                    print_results_predictions(last_predictions, train_words, targets, self.vocabulary_size)
                    print("")
