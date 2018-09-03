import pickle
from datetime import datetime
import argparse
import os
import tensorflow as tf
import math
import numpy as np
import time

from tools import *

from lstm import LSTM_Model
from word2vec import W2V_Model
from gen_cnn import Gen_CNN_Model


def create_placeholders(self):
    self.train_words = tf.placeholder(tf.int32, shape=[None, None], name="train_words")
    self.label_words = tf.placeholder(tf.int32, shape=[None, None], name="label_words")
    self.dropout = tf.placeholder(tf.float32, name="dropout")
    self.popularity_distribution_tensor = tf.tile(tf.reshape(tf.convert_to_tensor(self.popularity_distribution), [1,-1]), [self.batch_size, 1])

    self.context_words = tf.concat([tf.zeros([tf.shape(self.train_words)[0], self.seq_length], dtype=tf.float32), self.train_words])[:,:self.seq_length] if (self.model_params.D_type=='LSTM') \
        else tf.tile(self.train_words, tf.constant([1, self.seq_length]))[:,:self.seq_length]
    return self.train_words, self.label_words, self.dropout, self.popularity_distribution_tensor, self.context_words    

def create_generator(self):
    with tf.variable_scope("generator") as scope:
        if (self.model_params.G_type=='LSTM'):
            self.context_words = tf.concat([tf.zeros([tf.shape(self.train_words)[0], self.seq_length], dtype=tf.float32), self.train_words])[:,:self.seq_length]
        if (self.model_params.G_type=='CNN') or (self.model_params.G_type=='w2v'):
            self.context_words = tf.tile(self.train_words, tf.constant([1, self.seq_length]))[:,:self.seq_length]
        if (self.model_params.G_type=='LSTM'):
            self.generator = LSTM_Model(self)
        if (self.model_params.G_type=='CNN'):
            self.generator = Gen_CNN_Model(self)
        if (self.model_params.G_type=='w2v'):
            self.generator = W2V_Model(self)
        self.generator.create_embedding_layer()
        self.generator.embeddings_tensorflow = self.generator.embeddings(tf.constant(np.arange(self.vocabulary_size2), dtype=tf.int32))
        self.output_g = self.generator.creating_layer(self.context_words, self.dropout)
        
        self.g_loss1 = self.generator.compute_loss(self.label_words) if (self.model_params.G_type=='LSTM') \
            else self.generator.compute_loss(self.output_g, tf.reshape(self.label_words[:,-1], (-1,1)))
        self.before_softmax_G = self.generator.get_all_scores(self.output_g)
        self.output_distributions_G = self.generator.get_predictions(self.output_g)
        self.score_target_G = self.generator.get_score(self.output_g, tf.reshape(self.label_words[:,-1], [-1]))
        self.similarity_G = self.generator.get_similarity(self.context_words)
        self.g_weights = get_network_variables(self, "generator")
        self.mle_lossG = tf.reduce_sum(tf.log(tf.sigmoid(self.score_target_G)))
        
def create_discriminator(self):
    with tf.variable_scope("discriminator") as scope:
        if (self.model_params.D_type=='LSTM'):
            self.discriminator = LSTM_Model(self)
        if (self.model_params.D_type=='CNN'):
            self.discriminator = Gen_CNN_Model(self)
        if (self.model_params.D_type=='w2v'):
            self.discriminator = W2V_Model(self)
        self.discriminator.create_embedding_layer()
        self.discriminator.embeddings_tensorflow = self.discriminator.embeddings(tf.constant(np.arange(self.vocabulary_size2), dtype=tf.int32))
        self.output_d = self.discriminator.creating_layer(self.context_words, self.dropout)
        
        self.d_loss1 = self.discriminator.compute_loss(self.label_words) if (self.model_params.D_type=='LSTM') \
            else self.discriminator.compute_loss(self.output_d, tf.reshape(self.label_words[:,-1], (-1,1)))
        self.before_softmax_D = self.discriminator.get_all_scores(self.output_d)
        self.output_distributions_D = self.discriminator.get_predictions(self.output_d)
        self.similarity_D = self.discriminator.get_similarity(self.context_words)
        self.score_target_D = self.discriminator.get_score(self.output_d, tf.reshape(self.label_words[:,-1], [-1]))
        self.d_weights = get_network_variables(self, "discriminator")
        self.mle_lossD = tf.reduce_sum(tf.log(tf.sigmoid(self.score_target_D)))
        self.softmax_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels= tf.one_hot(tf.reshape(self.label_words[:,-1], [-1]), \
            self.vocabulary_size2), logits=self.before_softmax_D))


def get_adv_samples(self, argument, number):
    def sample_no_replacement(x):
        x = tf.nn.softmax(x)
        return tf.cast(tf.py_func(np.random.choice, [self.vocabulary_size2, number, False, x+1e-10], tf.int64), tf.int32)

    switcher = {
        "stochastic": tf.multinomial(self.before_softmax_G, num_samples=number, output_dtype=tf.int32, seed=self.model_params.seed),
        "no_stochastic": tf.reshape(tf.stack(tf.map_fn(sample_no_replacement, self.before_softmax_G, dtype=tf.int32)), [tf.shape(self.train_words)[0], number]),
        "argmax": tf.nn.top_k(self.before_softmax_G, k=number)[1]
    }
    return switcher.get(argument, "nothing")
def get_random_samples(self, argument, number):
    switcher = {
        "uniform": tf.random_uniform(shape=[tf.shape(self.train_words)[0], number], minval=0, maxval=self.vocabulary_size2-1, dtype=tf.int32, seed=self.model_params.seed),
        "one": tf.random_uniform(shape=[tf.shape(self.train_words)[0], number], minval=self.one_guy_sample, maxval=self.one_guy_sample+1, dtype=tf.int32, seed=self.model_params.seed),
        #"popularity": tf.multinomial(self.popularity_distribution_tensor, num_samples=number, output_dtype=tf.int32, seed=self.model_params.seed),
    }
    return switcher.get(argument, "nothing")

def discriminator_adversarial_loss(self):
    def fn(x):
        return x[0][x[1]]
    def fake_fn(fake_samples):
        elems = (self.before_softmax_D, tf.transpose(fake_samples))
        return tf.reshape(tf.stack(tf.map_fn(fn, elems, dtype=tf.float32)), [-1])
    def fake_fn_improved(elems):
        embedding, elem = elems[0], elems[1]
        nce_weights, nce_biases = tf.nn.embedding_lookup(self.discriminator.nce_weights, elem), \
                tf.nn.embedding_lookup(self.discriminator.nce_biases, elem)
        return tf.reduce_sum(tf.multiply(embedding, nce_weights), axis=1) + nce_biases

    #Get values for the targets
    #self.true_valuesD = tf.reshape(tf.stack(tf.map_fn(fn, (self.before_softmax_D, self.label_words[:,-1]), dtype=tf.float32, name="true_values")), [-1])
    self.true_valuesD = tf.reshape(tf.stack(tf.map_fn(fake_fn_improved, (self.output_d, tf.reshape(self.label_words[:,-1], (-1, 1))), dtype=tf.float32, name="true_values")), [-1])
    self.disc_self_samples = get_adv_samples(self, self.discriminator_samples_type[0], self.adv_negD)
    self.disc_random_samples = get_random_samples(self, self.discriminator_samples_type[1], self.random_negD)

    #Sampled Softmax loss
    #if (self.adv_discriminator_loss[0]=="softmax"):
        #true_elems_softmax = (self.output_distributions_D, self.label_words[:,-1]) #true_values_softmax = tf.log(tf.reshape(tf.stack(tf.map_fn(true_fn, true_elems_softmax, dtype=tf.float32, name="disc_values")), [-1])) #loss = tf.reduce_sum(true_values_softmax)
        #loss = -tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels= tf.one_hot(tf.reshape(self.label_words[:,-1], [-1]), self.vocabulary_size), logits=self.before_softmax_D))

    if ("SS" in self.adv_discriminator_loss[0]):
        self.disc_fake_samples = tf.concat([self.disc_self_samples, self.disc_random_samples, tf.reshape(self.label_words[:,-1], (-1, 1))], axis=1)
        #self.fake_values = tf.transpose(tf.stack(tf.map_fn(fake_fn, tf.transpose(self.disc_fake_samples), dtype=tf.float32, name="disc_fake_values")))
        self.fake_values = tf.stack(tf.map_fn(fake_fn_improved, (self.output_d, self.disc_fake_samples), dtype=tf.float32, name="disc_fake_values"))
        if (self.adv_discriminator_loss[0]=="SS"):
            loss = tf.reduce_sum(self.true_valuesD) - tf.reduce_sum(tf.log(tf.reduce_sum(tf.exp(self.fake_values), axis=1)))
        if (self.adv_discriminator_loss[0]=="SS_self"):
            loss = tf.reduce_sum(self.true_valuesD) - tf.reduce_sum(tf.reduce_mean(self.fake_values, axis=1))

    #Binary cross entropy as in GANs
    if (self.adv_discriminator_loss[0]=="BCE"):
        self.disc_fake_samples = tf.concat([self.disc_self_samples, self.disc_random_samples], axis=1)
        #self.fake_values = tf.transpose(tf.stack(tf.map_fn(fake_fn, tf.transpose(self.disc_fake_samples), dtype=tf.float32, name="disc_fake_values")))
        self.fake_values = tf.stack(tf.map_fn(fake_fn_improved, (self.output_d, self.disc_fake_samples), dtype=tf.float32, name="disc_fake_values"))
        loss = tf.reduce_sum(tf.log(tf.sigmoid(self.true_valuesD))) + tf.reduce_sum(tf.log(1-tf.sigmoid(self.fake_values)))
    
    if ("SS" in self.adv_discriminator_loss[0]) or (self.adv_discriminator_loss[0]=="BCE"):
        self.true_values_sigmoid, self.fake_values_sigmoid = tf.sigmoid(self.true_valuesD), tf.sigmoid(self.fake_values)
        
    return loss

def generator_adversarial_loss(self):
    def true_fn(x):
        return x[0][x[1]]
    def fn_G(fake_samples):
        elems = (self.before_softmax_G, tf.transpose(fake_samples))
        return tf.reshape(tf.stack(tf.map_fn(true_fn, elems, dtype=tf.float32)), [-1])
    def fn_D(fake_samples):
        elems = (self.before_softmax_D, tf.transpose(fake_samples))
        return tf.reshape(tf.stack(tf.map_fn(true_fn, elems, dtype=tf.float32)), [-1])
    def fn_D_improved(elems):
        embedding, items = elems[0], elems[1]
        nce_weights, nce_biases = tf.nn.embedding_lookup(self.discriminator.nce_weights, items), \
                tf.nn.embedding_lookup(self.discriminator.nce_biases, items)
        return tf.reduce_sum(tf.multiply(embedding, nce_weights), axis=1) + nce_biases
    def fn_G_improved(elems):
        embedding, items = elems[0], elems[1]
        nce_weights, nce_biases = tf.nn.embedding_lookup(self.generator.nce_weights, items), \
                tf.nn.embedding_lookup(self.generator.nce_biases, items)
        return tf.reduce_sum(tf.multiply(embedding, nce_weights), axis=1) + nce_biases

    #true_values = tf.reshape(tf.stack(tf.map_fn(true_fn, (self.before_softmax_G, self.label_words[:,-1]), dtype=tf.float32, name="true_values")), [-1])
    self.true_valuesG = tf.reshape(tf.stack(tf.map_fn(fn_G_improved, (self.output_g, tf.reshape(self.label_words[:,-1], (-1, 1))), dtype=tf.float32, name="true_values")), [-1])
    self.gen_self_samples = get_adv_samples(self, self.generator_samples_type[0], self.adv_negG)
    self.gen_random_samples = get_random_samples(self, self.generator_samples_type[1], self.random_negG)
    self.gen_true_samples = tf.reshape(self.label_words[:,-1], (-1, 1))
    self.gen_fake_samples = tf.concat([self.gen_self_samples, self.gen_random_samples, self.gen_true_samples], axis=1)

    fake_elems = (self.output_g, self.gen_fake_samples)
    self.gen_fake_values = tf.stack(tf.map_fn(fn_G_improved, fake_elems, dtype=tf.float32, name="true_values")) 
    #self.gen_fake_values = tf.transpose(tf.stack(tf.map_fn(fn_G, tf.transpose(self.gen_fake_samples), dtype=tf.float32, name="gen_fake_values")))

    if (self.adv_generator_loss[0]=="MALIGAN"):
        self.gen_fake_samples = tf.concat([self.gen_self_samples, self.gen_random_samples], axis=1)
        self.log_p_g = tf.log(tf.sigmoid(tf.transpose(tf.stack(tf.map_fn(fn_G, tf.transpose(self.gen_fake_samples), dtype=tf.float32)))))
        self.p_d = tf.sigmoid(tf.transpose(tf.stack(tf.map_fn(fn_D, tf.transpose(self.gen_fake_samples), dtype=tf.float32))))
        self.r_d = self.p_d/(1-self.p_d)
        self.r_d = (self.r_d/tf.reduce_sum(self.r_d))
        gen_loss = tf.reduce_sum(tf.multiply(self.log_p_g, self.r_d))

    if (self.adv_generator_loss[0]=="ADV_NS"):
        gen_loss = tf.reduce_sum(self.true_valuesG) - tf.reduce_sum(tf.log(tf.reduce_sum(tf.exp(self.gen_fake_values), axis=1)))

    if ("ADV_IS" in self.adv_generator_loss[0]=="ADV_IS") :        
        def get_gen_fake_values_D(argument):
            switcher = {
                "ADV_IS": tf.sigmoid(-self.gen_fake_values_D), #tf.nn.softmax(-gen_fake_values_D),
                "Inverse_ADV_IS": tf.nn.softmax(self.gen_fake_values_D),
                "Random_ADV_IS": tf.random_uniform([tf.shape(self.gen_fake_values)[0], tf.shape(self.gen_fake_values)[1]], minval=0, maxval=1),
                "Uniform_ADV_IS": tf.ones([tf.shape(self.gen_fake_values)[0], tf.shape(self.gen_fake_values)[1]]),
                "Normal_ADV_IS":  self.gen_fake_values_D
            }
            return switcher.get(argument, "nothing")
        self.gen_fake_values_D = tf.stack(tf.map_fn(fn_D_improved, (self.output_d, self.gen_fake_samples), dtype=tf.float32, name="true_values"))
        #gen_fake_values_D = tf.transpose(tf.stack(tf.map_fn(fn_D, tf.transpose(self.gen_fake_samples), dtype=tf.float32, name="gen_fake_values_D")))
        self.gen_fake_values_D = get_gen_fake_values_D(self.adv_generator_loss[0])
        #gen_loss = tf.reduce_sum(self.true_valuesG) - tf.reduce_sum(tf.multiply(self.gen_fake_values, gen_fake_values_D))
        
        gen_loss = tf.reduce_sum(self.true_valuesG) - tf.reduce_sum(tf.log(tf.reduce_sum(tf.multiply(tf.exp(self.gen_fake_values), self.gen_fake_values_D), axis=1)))
    return gen_loss

def get_network_variables(self, name_of_the_network):
        train_variables = tf.trainable_variables()
        return [v for v in train_variables if v.name.startswith(name_of_the_network)]