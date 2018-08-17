import numpy as np
from tqdm import tqdm
import tensorflow as tf
from keras.datasets import imdb
import copy
from keras.models import Sequential, Model
from keras.utils import to_categorical
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Input, Masking, Conv1D, MaxPooling1D
from keras.layers import Reshape,Conv2D, MaxPooling2D, Flatten, Concatenate
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras import losses
from random import shuffle

from metric_utils import *
import math

class Gen_CNN_Model(object):

    def __init__(self, model):
        self.model_params = model
        self.vocabulary_size, self.vocabulary_size2 = model.vocabulary_size, model.vocabulary_size2
        self.embedding_size = model.embedding_size
        self.batch_size = model.batch_size
        self.seq_length = model.seq_length
        self.training_data, self.test_data, self.test_list_batches = model.training_data, model.test_data, model.test_list_batches
        self.neg_sampled = 50
        self.num_units = 64
        self.num_epochs = model.epoch
        self.learning_rate = 1e-3
        self.printing_step = 100
        self.use_pretrained_embeddings = False

        self.number_of_convolutions = 2
        self.num_filters = 100
        self.filter_sizes = [2, 3, 5, 10]
        self.max_pooling_window = 4
        self.second_filter_size = 3
        self.num_second_filters = 100

    def create_embedding_layer(self):
        with tf.name_scope("model"):
            if (self.use_pretrained_embeddings):
                print("Initializing with the previous embedding matrix")
                self.embeddings = Embedding(self.vocabulary_size, self.embedding_size, weights=[self.embedding_matrix], input_length=self.seq_length, trainable=True)
            else:
                print("Initializing with a random embedding matrix")
                self.embeddings = Embedding(self.vocabulary_size, self.embedding_size, embeddings_initializer='uniform', input_length=self.seq_length)
            return self.embeddings

    def convolve_4d_matrix(self, input_matrix, filter_shape, filter_size, num_filters, max_pooling, pooling_window_size):
        bias = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b-%s" % filter_size,trainable=True)
        self.filter = tf.Variable(tf.truncated_normal(filter_shape,mean= 0.0, stddev=0.1), name="filter-%s" % filter_size,trainable=True)
        
        conv1 = tf.nn.conv2d(input_matrix, self.filter, strides=[1, 1, 1, 1], padding="VALID", name="conv1")
        h1 = tf.nn.relu(tf.nn.bias_add(conv1, bias), name="relu1")
        if (max_pooling):
            pooled = tf.nn.max_pool(h1, ksize= [1, pooling_window_size, 1, 1], strides=[1,pooling_window_size,1,1], padding='VALID', name="pool1")
        else:
            pooled = h1
        return pooled

    def creating_layer(self, input_tensor, dropout):
        embed = self.embeddings(input_tensor)
        self.embedded_chars_expanded = tf.expand_dims(embed, -1)
        pooled_outputs = []
        with tf.name_scope("conv-maxpool"):
            for i, filter_size in enumerate(self.filter_sizes):
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                pooled = self.convolve_4d_matrix(self.embedded_chars_expanded, filter_shape, filter_size, self.num_filters, \
                    max_pooling=True, pooling_window_size=self.max_pooling_window)
                
                if (self.number_of_convolutions==2):
                    filter_shape = [self.second_filter_size, self.num_filters, 1, self.num_second_filters]
                    pooled = self.convolve_4d_matrix(tf.transpose(pooled, perm=[0,1,3,2]), filter_shape, self.second_filter_size, \
                        self.num_second_filters, max_pooling=False, pooling_window_size=0)
                
                pooled = tf.reduce_max(pooled, axis=1, keepdims=True)
                pooled_outputs.append(pooled)

        self.h_pool1 = tf.concat(pooled_outputs, 3)
        self.h_drop1 = tf.reshape(self.h_pool1, [-1, self.num_filters * len(self.filter_sizes)]) if (self.number_of_convolutions==1) \
            else tf.reshape(self.h_pool1, [-1, self.num_second_filters * len(self.filter_sizes)])
        self.h_drop1 = tf.nn.dropout(self.h_drop1, dropout)
        return self.h_drop1

    def averaging_embeddings(self, embed):
        embeddings_averaged = tf.reduce_mean(embed, axis=1)
        return embeddings_averaged

    def compute_loss(self, output, label_tensor):
        with tf.name_scope("output_layer"):
            self.nce_weights = tf.Variable(tf.truncated_normal([self.vocabulary_size2, self.num_filters * len(self.filter_sizes)], stddev=1.0 / math.sqrt(self.embedding_size)))
            self.nce_biases = tf.Variable(tf.zeros([self.vocabulary_size2]))
            self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
                weights=self.nce_weights, biases=self.nce_biases, labels=label_tensor,
                inputs=output, num_sampled=self.neg_sampled, num_classes=self.vocabulary_size2))
        return self.loss

    def create_graph(self):
        self.train_inputs = tf.placeholder(tf.int32, shape=[None, None], name="train_inputs")
        self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name="train_labels")
        self.dropout = tf.placeholder(tf.float32, name="dropout")

        self.create_embedding_layer()
        self.output = self.creating_layer(self.train_inputs)
        self.loss = self.compute_loss(self.output, self.train_labels)
        with tf.name_scope('optimizer'):
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.optimizer = tf.train.AdamOptimizer(1e-3, beta1=0.8 ,beta2= 0.9, epsilon=1e-5).minimize(self.loss)
            optimizer_plot = tf.train.GradientDescentOptimizer(self.learning_rate)
            grads_and_vars = optimizer_plot.compute_gradients(self.loss)
            self.train_op = optimizer_plot.apply_gradients(grads_and_vars, global_step=self.global_step)

    def get_predictions(self, output):
        self.last_predictions = tf.nn.softmax(tf.matmul(output,tf.transpose(self.nce_weights))+self.nce_biases)
        return self.last_predictions
    def get_all_scores(self, output):
        self.before_softmax = tf.matmul(output,tf.transpose(self.nce_weights))+self.nce_biases
        return self.before_softmax
    def get_score(self, output, elem):
        input_tensor = tf.reshape(elem, [-1])
        nce_weights = tf.nn.embedding_lookup(self.nce_weights, elem)
        nce_biases = tf.nn.embedding_lookup(self.nce_biases, elem)
        return tf.matmul(output,tf.transpose(nce_weights)) + nce_biases
    def get_similarity(self, input_tensor):
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings_tensorflow), 1, keepdims=True))
        normalized_embeddings = self.embeddings_tensorflow / norm
        output = tf.reduce_mean(tf.nn.embedding_lookup(normalized_embeddings, input_tensor), axis=1)
        self.similarity = tf.matmul(output, tf.transpose(normalized_embeddings))
        return self.similarity

    def initialize_model(self):
        self.create_graph()
        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())
        self.MPR, self.p1, self.p1p, self.step = list(), list(), list(), 0

    def training_step(self):
        train_words, label_words = create_batch(self)
        training_step(self, [self.optimizer, self.loss])

        if (self.step % self.printing_step == 0):
            if (self.model_params.metric=="MPR"):
                print_confidence_intervals(self, "selfplay")

            if (self.model_params.print_params[0]) and (self.model_params.type_of_data=="synthetic"):
                true_neg_prop, auc_score = print_candidate_sampling(self, self.model_params.print_params[1:])
                self.true_neg_prop.append(true_neg_prop)
                self.auc_score.append(auc_score)

            if (self.model_params.metric=="Analogy"):
                check_analogy(self)
                np.save("nlp/"+self.model_params.save_params[1]+"_embeddings", self._sess.run([self.discriminator.embeddings_tensorflow])[0])