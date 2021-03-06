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
    
    #if (self.model_params.sampling=="context"):
    #    self.conditional_distributions_tensor = tf.Variable(self.model_params.conditional_distributions, trainable=False)
    #    self.co_occurences_tensor = tf.Variable(self.list_co_occurences, trainable=False)
    #else:
    #    self.conditional_distributions_tensor, self.co_occurences_tensor = tf.Variable(np.ones(shape=(10,10))), tf.Variable(np.ones(shape=(10,10)))

    if (self.model_params.D_type=='LSTM') and (self.model_params.G_type=='LSTM'):
        self.train_words = tf.concat([tf.zeros([tf.shape(self.train_words)[0], self.seq_length], dtype=tf.float32), self.train_words])[:,:self.seq_length]
    if (self.model_params.D_type=='CNN') or (self.model_params.G_type=='CNN'):
        self.train_words = tf.tile(self.train_words, tf.constant([1, self.seq_length]))[:,:self.seq_length]

def create_generator(self, size):
    with tf.variable_scope("generator") as scope:
        if (self.model_params.G_type=='LSTM'):
            self.generator = LSTM_Model(self)
        if (self.model_params.G_type=='CNN'):
            self.generator = Gen_CNN_Model(self)
        if (self.model_params.G_type=='w2v'):
            self.generator = W2V_Model(self, size)

        self.generator.create_embedding_layer()
        self.generator.embeddings_tensorflow = self.generator.embeddings(tf.constant(np.arange(self.vocabulary_size), dtype=tf.int32))
        self.context_emb_g = self.generator.creating_layer(self.train_words, self.dropout)
        self.target_emb_g = self.generator.creating_layer(self.label_words, self.dropout)
        
        self.g_loss1 = self.generator.compute_loss(self.label_words) if (self.model_params.G_type=='LSTM') \
            else self.generator.compute_loss(self.context_emb_g, self.label_words)
        self.before_softmax_G = self.generator.get_all_scores(self.context_emb_g)
        self.output_distributions_G = self.generator.get_predictions(self.context_emb_g)

        self.score_target_G = self.generator.get_score(self.context_emb_g, tf.reshape(self.label_words, [-1]))
        self.similarity_G = self.generator.get_similarity(self.train_words)
        self.g_weights = get_network_variables(self, "generator")
        self.mle_lossG = tf.reduce_sum(tf.log(tf.sigmoid(self.score_target_G)))
        
def create_discriminator(self, size):
    with tf.variable_scope("discriminator") as scope:
        if (self.model_params.D_type=='LSTM'):
            self.discriminator = LSTM_Model(self)
        if (self.model_params.D_type=='CNN'):
            self.discriminator = Gen_CNN_Model(self)
        if (self.model_params.D_type=='w2v'):
            self.discriminator = W2V_Model(self, size)

        self.discriminator.create_embedding_layer()
        self.discriminator.embeddings_tensorflow = self.discriminator.embeddings(tf.constant(np.arange(self.vocabulary_size), dtype=tf.int32))
        self.context_emb_d = self.discriminator.creating_layer(self.train_words, self.dropout)
        self.target_emb_d = self.discriminator.creating_layer(self.label_words, self.dropout)
        
        self.d_loss1 = self.discriminator.compute_loss(self.context_emb_d, self.label_words)
        self.before_softmax_D = self.discriminator.get_all_scores(self.context_emb_d)
        self.output_distributions_D = self.discriminator.get_predictions(self.context_emb_d)
        
        self.score_target_D = self.discriminator.get_score(self.context_emb_d, tf.reshape(self.label_words, [-1]))
        self.similarity_D = self.discriminator.get_similarity(self.train_words)
        self.mle_lossD = tf.reduce_sum(tf.log(tf.sigmoid(self.score_target_D)))
        self.softmax_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels= tf.one_hot(tf.reshape(self.label_words[:,-1], [-1]), \
            self.vocabulary_size2), logits=self.before_softmax_D))
        self.d_weights = get_network_variables(self, "discriminator")


def get_adv_samples(self, argument, number):
    SS_size = 100
    seed = self.model_params.seed
    dist_selfplay = self.before_softmax_G if (self.model_params.model_type=="AIS") else self.before_softmax_D
    dist_context = self.generator.get_similarity(self.train_words) if (self.model_params.model_type=="AIS") else self.discriminator.get_similarity(self.train_words)
    dist_target = self.generator.get_similarity(self.label_words) if (self.model_params.model_type=="AIS") else self.discriminator.get_similarity(self.label_words)
    switcher = {
        "selfplay": tf.multinomial(dist_selfplay, num_samples=number, output_dtype=tf.int32, seed=seed),
        "not_selfplay": tf.multinomial(-dist_selfplay, num_samples=number, output_dtype=tf.int32, seed=seed),
        "top_selfplay": tf.cast(tf.nn.top_k(-dist_selfplay, k=number)[1], dtype=tf.int32),
        "context_emb": tf.multinomial(dist_context, num_samples=number, output_dtype=tf.int32, seed=seed),
        "target_emb": tf.multinomial(dist_target, num_samples=number, output_dtype=tf.int32, seed=seed),
        "uniform": tf.random_uniform(shape=[tf.shape(self.train_words)[0], number], minval=0, maxval=self.vocabulary_size2-1, dtype=tf.int32, seed=seed),
        "top_then_selfplay": get_adv_samples_subset_method(self, "selfplay", SS_size, number),
        "top_then_uniform": get_adv_samples_subset_method(self, "uniform", SS_size, number)
    }
    return switcher.get(argument)


def get_adv_samples_subset_method(self, argument, SS_size, number):
    def fn(elem):
        return tf.stack(tf.map_fn(lambda x : elem[0][x], elem[1], dtype=tf.int32))
    def fn2(elem):
        return tf.stack(tf.map_fn(lambda x : elem[0][x], elem[1], dtype=tf.float32))
    seed = self.model_params.seed
    dist_selfplay = self.before_softmax_G if (self.model_params.model_type=="AIS") else self.before_softmax_D
    dist_context = self.generator.get_similarity(self.train_words) if (self.model_params.model_type=="AIS") else self.discriminator.get_similarity(self.train_words)
    dist_target = self.generator.get_similarity(self.label_words) if (self.model_params.model_type=="AIS") else self.discriminator.get_similarity(self.label_words)
    subset_samples = tf.cast(tf.nn.top_k(dist_selfplay, k=SS_size)[1], dtype=tf.int32)
    if (argument=="selfplay"):
        subset_values = tf.stack(tf.map_fn(fn2, (dist_selfplay, subset_samples), dtype=tf.float32))
        final_samples = tf.multinomial(subset_values, num_samples=number, output_dtype=tf.int32, seed=seed)
    else:
        final_samples = tf.random_uniform(shape=[tf.shape(self.train_words)[0], number], minval=0, maxval=SS_size, dtype=tf.int32, seed=seed),
    final_samples = tf.stack(tf.map_fn(fn, (subset_samples, final_samples), dtype=tf.int32))
    return final_samples


def get_adv_samples_for_whole_batch(self, argument, number):
    seed = self.model_params.seed
    switcher = {
        "selfplay": tf.multinomial(self.before_softmax_G[0:1], num_samples=number, output_dtype=tf.int64, seed=seed),
        "top_selfplay": tf.cast(tf.nn.top_k(self.before_softmax_G[0], k=number)[1], dtype=tf.int32),
        "not_selfplay": tf.multinomial(-self.before_softmax_G[0:1], num_samples=number, output_dtype=tf.int32, seed=seed),
        "uniform": tf.random_uniform(shape=[number], minval=0, maxval=self.vocabulary_size2-1, dtype=tf.int32, seed=seed),
        "not_emp": tf.multinomial(tf.nn.embedding_lookup(self.conditional_distributions_tensor, self.train_words[0]), num_samples=number, output_dtype=tf.int32, seed=seed),
        "emp": tf.multinomial(tf.nn.embedding_lookup(-self.conditional_distributions_tensor, self.train_words[0]), num_samples=number, output_dtype=tf.int32, seed=seed)
    }
    return switcher.get(argument)


def sampled_softmax_loss_improved(self):
    def fn(elem):
        return self.output_distributions_D[0][elem]
    self.disc_ss = tf.reshape(get_adv_samples_for_whole_batch(self, self.model_params.sampling, self.negD), [-1])
    self.disc_tv = tf.reshape(tf.stack(tf.map_fn(fn, self.label_words[:,0], dtype=tf.float32)), (-1,1))
    self.disc_sv = tf.stack(tf.map_fn(fn, self.disc_ss, dtype=tf.float32))
    loss = tf.reduce_sum(tf.nn.sampled_softmax_loss(
        weights = self.discriminator.weights, biases = self.discriminator.biases,
        labels = self.label_words, inputs = self.context_emb_d, num_sampled = self.negD,
        num_classes = self.model_params.vocabulary_size2,
        sampled_values = (self.disc_ss, self.disc_tv, self.disc_sv),
        seed = self.model_params.seed))
    return loss


def discriminator_adversarial_loss(self):
    self.disc_self_samples = tf.concat([get_adv_samples(self, sampling, self.negD) for sampling in self.model_params.sampling], axis=1) if (len(self.model_params.sampling)>1) else \
        get_adv_samples(self, self.model_params.sampling[0], self.negD)
    self.disc_fake_samples = tf.concat([self.disc_self_samples, self.label_words], axis=1) if ("SS" in self.adv_discriminator_loss[0]) else \
        tf.concat([self.disc_self_samples], axis=1)
    self.disc_fake_values = tf.reduce_sum(tf.multiply(tf.expand_dims(self.context_emb_d, 1), \
        tf.nn.embedding_lookup(self.discriminator.weights, self.disc_fake_samples)), axis=-1) + tf.nn.embedding_lookup(self.discriminator.biases, self.disc_fake_samples)
    
    loss = tf.reduce_sum(self.score_target_D) - tf.reduce_sum(tf.log(tf.reduce_sum(tf.exp(self.disc_fake_values), axis=1))) if ("SS" in self.adv_discriminator_loss[0]) else \
        tf.reduce_sum(tf.log(tf.sigmoid(self.score_target_D))) + tf.reduce_sum(tf.log(1-tf.sigmoid(self.disc_fake_values)))
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
        weights, biases = tf.nn.embedding_lookup(self.discriminator.weights, items), tf.nn.embedding_lookup(self.discriminator.biases, items)
        return tf.reduce_sum(tf.multiply(embedding, weights), axis=1) + biases
    def fn_G_improved(elems):
        embedding, items = elems[0], elems[1]
        weights, biases = tf.nn.embedding_lookup(self.generator.weights, items), tf.nn.embedding_lookup(self.generator.biases, items)
        return tf.reduce_sum(tf.multiply(embedding, weights), axis=1) + biases

    self.gen_self_samples = tf.concat([get_adv_samples(self, sampling, self.negD) for sampling in self.model_params.sampling], axis=1) if (len(self.model_params.sampling)>1) else \
        get_adv_samples(self, self.model_params.sampling[0], self.negD)

    if (self.model_params.model_type=="MALIGAN"):
        self.log_p_g = tf.log(tf.sigmoid(tf.transpose(tf.stack(tf.map_fn(fn_G, tf.transpose(self.gen_self_samples), dtype=tf.float32)))))
        self.p_d = tf.sigmoid(tf.transpose(tf.stack(tf.map_fn(fn_D, tf.transpose(self.gen_self_samples), dtype=tf.float32))))
        self.r_d = self.p_d/(1-self.p_d)
        self.r_d = (self.r_d/tf.reduce_sum(self.r_d))
        gen_loss = tf.reduce_sum(tf.multiply(self.log_p_g, self.r_d))

    if ("AIS" in self.model_params.model_type):
        self.gen_fake_samples = tf.concat([self.gen_self_samples, self.label_words], axis=1) if (self.model_params.model_type=="AIS") else self.gen_self_samples
        self.gen_fake_values = tf.stack(tf.map_fn(fn_G_improved, (self.context_emb_g, self.gen_fake_samples), dtype=tf.float32, name="gen_fake_values")) 
        
        self.gen_fake_values_D = tf.stack(tf.map_fn(fn_D_improved, (self.context_emb_d, self.gen_fake_samples), dtype=tf.float32, name="gen_fake_values_D"))
        self.gen_self_values_D_sigmoid_aux = tf.sigmoid(tf.stack(tf.map_fn(fn_D_improved, (self.context_emb_d, self.gen_self_samples), dtype=tf.float32, name="gen_self_values_D")))
        self.gen_fake_values_D_sigmoid, self.gen_fake_values_D_softmax = tf.sigmoid(self.gen_fake_values_D), tf.nn.softmax(self.gen_fake_values_D)
        self.gen_self_values_D_sigmoid, self.gen_self_values_D_softmax = self.gen_fake_values_D_sigmoid[:,:-1], self.gen_fake_values_D_softmax[:,:-1]
        self.gen_true_values_D_sigmoid, self.gen_true_values_D_softmax = self.gen_fake_values_D_sigmoid[:,-1], self.gen_fake_values_D_softmax[:,-1]

        #self.gen_fake_values_D = tf.sigmoid(-self.gen_fake_values_D)
        self.gen_fake_values_D = tf.nn.softmax(-self.gen_fake_values_D)
        
        if (self.model_params.model_type=="AIS"):
            gen_loss = tf.reduce_sum(self.score_target_G) + tf.reduce_sum(tf.log(self.gen_fake_values_D[:,-1])) - tf.reduce_sum(tf.log(tf.reduce_sum(tf.multiply(tf.exp(self.gen_fake_values), self.gen_fake_values_D), axis=1)))
        if (self.model_params.model_type=="AIS-BCE"):
            gen_loss = tf.reduce_sum(self.score_target_G) + tf.reduce_sum(tf.multiply(tf.log(1-tf.sigmoid(self.gen_fake_values)), self.gen_fake_values_D))
    return gen_loss


def sampled_softmax_loss_improved_gen(self):
    self.gen_ss = tf.reshape(get_adv_samples_for_whole_batch(self, self.model_params.sampling, self.negG), [-1])
    self.gen_sv = tf.tile(tf.reshape(self.generator.get_score(tf.tile(self.context_emb_g[:1], tf.constant([self.negG,1])), self.gen_ss), (1, -1)), tf.shape(self.train_words[:,:1]))
    self.gen_tv = tf.reshape(self.score_target_G, (-1,1))
    self.gen_fv = tf.concat([self.gen_sv, self.gen_tv], axis=1)
    self.gen_sv_disc = tf.tile(tf.reshape(self.discriminator.get_score(tf.tile(self.context_emb_d[:1], tf.constant([self.negG,1])), self.gen_ss), (1, -1)), tf.shape(self.train_words[:,:1]))
    self.gen_tv_disc = tf.reshape(self.score_target_D, (-1,1))
    self.gen_fv_disc = tf.concat([self.gen_sv_disc, self.gen_tv_disc], axis=1)
    self.gen_fv_disc = tf.nn.sigmoid(-self.gen_fv_disc)    
    loss = tf.reduce_sum(self.score_target_G) + tf.reduce_sum(tf.log(self.gen_fv_disc[:,-1])) - tf.reduce_sum(tf.log(tf.reduce_sum(tf.multiply(tf.exp(self.gen_fv), self.gen_fv_disc), axis=1)))
    return loss


def get_network_variables(self, name_of_the_network):
        train_variables = tf.trainable_variables()
        return [v for v in train_variables if v.name.startswith(name_of_the_network)]