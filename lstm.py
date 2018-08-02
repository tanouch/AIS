import numpy as np
from tqdm import tqdm
import tensorflow as tf
from keras.datasets import imdb
import copy
from keras.models import Sequential, Model
from keras.utils import to_categorical
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Input, Masking, Conv1D, MaxPooling1D, Reshape,Conv2D, MaxPooling2D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras import losses
from random import shuffle

import math
from tools import *
from metric_utils import *

tf.set_random_seed(42)

class LSTM_Model(object):

    def __init__(self, wider_model):
        self.vocabulary_size = wider_model.vocabulary_size
        self.embedding_size = wider_model.embedding_size
        self.batch_size = wider_model.batch_size
        self.seq_length = wider_model.seq_length
        self.num_epochs = wider_model.epoch
        self.embedding_matrix = wider_model.embedding_matrix
        self.use_pretrained_embeddings = wider_model.use_pretrained_embeddings
        self.neg_sampled = 50
        self.num_units = 64
        self.learning_rate = 1e-2
        self.printing_step = 50
        self.X_train, self.Y_train, self.X_test, self.Y_test, self.LSTM_labels_train, self.LSTM_labels_test = list(), list(), list(), list(), list(), list()
        self.model_type = "recurrent"


    def create_embedding_layer(self):
        with tf.name_scope("embeddings"):
            if not self.use_pretrained_embeddings :
                print("Initializing with a random embedding matrix")
                self.W = Embedding(self.vocabulary_size+1, self.embedding_size, embeddings_initializer='uniform', input_length=self.seq_length, mask_zero=True)
            else:
                print("Initializing with the previous embedding matrix")
                self.W = Embedding(self.vocabulary_size+1, self.embedding_size, weights=[self.embedding_matrix], input_length=self.seq_length, trainable=True, mask_zero=True)

        return self.W

    def creating_layer(self, input_tensor, dropout):
        with tf.name_scope("model"):
            layer1 = self.W(input_tensor)
            if (self.model_type=="recurrent"):
                layer2 = Bidirectional(LSTM(self.embedding_size, return_sequences=True))(layer1)
                layer3 = tf.nn.dropout(layer2, dropout)
                final_lstm = LSTM(self.embedding_size, return_sequences=True)(layer3)
                final_lstm = tf.nn.dropout(final_lstm, dropout)
                self.output = tf.unstack(final_lstm, num=self.seq_length, axis=1)
                return self.output

            elif (self.model_type=="deep_sets"):
                averaging_layer1 = tf.reduce_sum(layer1, axis=1, keep_dims=True)
                final_layer1 = tf.add(0.5*layer1, 0.5*averaging_layer1)

                layer2 = Bidirectional(LSTM(self.embedding_size, return_sequences=True))(final_layer1)
                layer3 = tf.nn.dropout(layer2, dropout)
                averaging_layer3 = tf.reduce_sum(layer3, axis=1, keep_dims=True)
                final_layer3 = tf.add(0.5*layer3, 0.5*averaging_layer3)

                layer4 = Bidirectional(LSTM(self.embedding_size, return_sequences=True))(final_layer3)
                layer5 = tf.nn.dropout(layer4, dropout)
                averaging_layer5 = tf.reduce_sum(layer5, axis=1, keep_dims=True)
                final_layer5 = tf.add(0.5*layer5, 0.5*averaging_layer5)

                layer6 = LSTM(self.embedding_size, return_sequences=True)(final_layer5)
                averaging_layer6 = tf.reduce_sum(layer6, axis=1, keep_dims=True)
                final_layer6 = tf.add(0.5*layer6, 0.5*averaging_layer6)
                self.output = tf.unstack(final_layer6, num=self.seq_length, axis=1)
                return self.output

            elif (self.model_type=="deep_sets_real"):
                layer2 = Bidirectional(LSTM(self.embedding_size, return_sequences=True))(layer1)
                layer3 = tf.nn.dropout(layer2, dropout)
                layer6 = Bidirectional(LSTM(self.embedding_size, return_sequences=True))(layer3)
                layer7 = tf.nn.dropout(layer6, dropout)
                final_lstm = LSTM(self.embedding_size, return_sequences=True)(layer7)
                averaged_output = tf.reduce_mean(final_lstm, axis=1)
                return self.output

            elif (self.model_type=="averaging"):
                averaging_layer = tf.reduce_mean(layer1, axis=1, keep_dims=True)
                layer2 = LSTM(self.embedding_size, return_sequences=True)(averaging_layer)
                layer3 = tf.nn.dropout(layer2, dropout)
                layer6 = LSTM(self.embedding_size, return_sequences=False)(layer3)
                self.output = tf.nn.dropout(layer6, dropout)
                return self.output


    def compute_loss(self, label_tensor):
        with tf.name_scope("output_layer"):
            self.weights = tf.Variable(tf.truncated_normal([self.vocabulary_size+1, self.embedding_size], stddev=1.0 / math.sqrt(self.embedding_size)), name="weights")
            self.biases = tf.Variable(tf.zeros([self.vocabulary_size+1]), name="biases")

        with tf.name_scope("loss"):
            if (self.model_type == 'recurrent') or (self.model_type=='deep_sets'):
                self.step_losses = []
                for t in range(self.seq_length):
                    output_t = self.output[t]
                    label_t = tf.reshape(label_tensor[:,t],[-1,1])
                    loss = tf.reduce_mean(tf.nn.nce_loss(
                        weights=self.weights, biases=self.biases, labels=label_t,
                        inputs=output_t, num_sampled=self.neg_sampled, num_classes=self.vocabulary_size+1))
                    self.step_losses.append(loss)

            if (self.model_type=="averaging") or (self.model_type=="deep_sets_real"):
                self.step_losses = [tf.reduce_mean(tf.nn.nce_loss(
                    weights=self.weights, biases=self.biases, labels=tf.reshape(label_tensor[:,self.seq_length-1],[-1,1]),
                    inputs=self.output, num_sampled=self.neg_sampled, num_classes=self.vocabulary_size+1))]

            self.loss = tf.reduce_mean(self.step_losses)
        return self.loss

    def get_predictions(self, output):
        return tf.nn.softmax(tf.matmul(output, tf.transpose(self.weights)) + self.biases)

    def get_predictions_averaging(self, output):
        return tf.nn.softmax(tf.matmul(tf.reduce_mean(output, axis=0, keep_dims=True), tf.transpose(self.weights)) + self.biases)

    def create_graph(self):
        with tf.name_scope("inputs"):
            self.train_words = tf.placeholder(tf.int32, shape=[None, self.seq_length])
            self.label_words = tf.placeholder(tf.int32, shape=[None, self.seq_length])
            self.dropout = tf.placeholder(tf.float32)

        self.create_embedding_layer()
        self.output = self.creating_layer(self.train_words, self.dropout)
        self.compute_loss(self.label_words)
        if (self.model_type=="recurrent") or (self.model_type=="deep_sets"):
            self.last_predictions = self.get_predictions(self.output[-1])
            self.last_predictions_averaging = self.get_predictions_averaging(self.output[-1])
        else:
            self.last_predictions = self.get_predictions(self.output)

        with tf.name_scope('optimizer'):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            optimizer_plot = tf.train.AdamOptimizer(self.learning_rate)
            grads_and_vars = optimizer_plot.compute_gradients(self.loss)
            self.train_op = optimizer_plot.apply_gradients(grads_and_vars, global_step=self.global_step)

    def train_model_with_tensorflow(self):
        self.create_graph()
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())
        total_loss, step, epoch = 0, 0, 0
        steps_per_epoch = int(len(self.X_train)/self.batch_size)

        while (epoch < self.num_epochs):
            batch_inputs, batch_labels = get_the_next_batch_random(self.X_train, self.LSTM_labels_train, self.batch_size)

            _, loss = self._sess.run([self.optimizer, self.loss], \
                feed_dict={self.train_words:batch_inputs, self.label_words:batch_labels, self.dropout:1})
            total_loss += loss
            step += 1
            epoch = epoch + 1 if (step % steps_per_epoch ==0) else epoch

            if (step % self.printing_step == 0) or (step==1):
                print("Epoch "+ str(epoch) + " step "+ str(step) +" and loss " + str(total_loss/self.printing_step))
                total_loss = 0
                batch_inputs, batch_labels = get_the_next_batch_random(self.X_test, self.Y_test, len(self.X_test)-1)
                last_predictions = self._sess.run([self.last_predictions], feed_dict={self.train_words:batch_inputs, self.dropout:1.0})[0]
                print_results_predictions(last_predictions, batch_inputs, batch_labels, self.vocabulary_size)

                #last_predictions = list()
                #for basket in tqdm(batch_inputs):
                #    batch_inputs_aux = list_of_shuffle_batches(basket)
                #    last_predictions.append(self._sess.run([self.last_predictions_averaging], feed_dict={self.train_words:batch_inputs_aux, self.dropout:1.0})[0].flatten())
                #print(last_predictions[0].shape)
                #print_results_predictions(last_predictions, batch_inputs, batch_labels, self.vocabulary_size)
                print("")
