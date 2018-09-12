"""Basic word2vec example."""
import collections
import math
import os
import sys
import argparse
import random
from tempfile import gettempdir
import zipfile
import numpy as np
from six.moves import urllib
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def read_data_text8(filename):
  """Extract the first file enclosed in a zip file as a list of words."""
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data

def build_dataset(words, n_words, top_words_removed_threshold):
  """Process raw inputs into a dataset."""
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(n_words - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words:
    index = dictionary.get(word, 0)
    if index == 0:  # dictionary['UNK']
      unk_count += 1
    elif (index <= top_words_removed_threshold):
      unk_count += 1
    else:
      data.append(index)
  count[0][1] = unk_count
  reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reversed_dictionary

def generate_batch_google(self, num_skips, skip_window):
  data = self.model_params.training_data
  assert self.model_params.batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window

  batch = np.ndarray(shape=(self.model_params.batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(self.model_params.batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
  if self.model_params.data_index + span > len(data):
    self.model_params.data_index = 0
  buffer.extend(data[self.model_params.data_index:self.model_params.data_index + span])
  self.model_params.data_index += span
  for i in range(self.model_params.batch_size // num_skips):
    context_words = [w for w in range(span) if w != skip_window]
    words_to_use = random.sample(context_words, num_skips)
    for j, context_word in enumerate(words_to_use):
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[context_word]
    if self.model_params.data_index == len(data):
      buffer.extend(data[0:span])
      self.model_params.data_index = span
    else:
      buffer.append(data[self.model_params.data_index])
      self.model_params.data_index += 1
  self.model_params.data_index = (self.model_params.data_index + len(data) - span) % len(data)
  batches = np.column_stack((batch.flatten(), labels.flatten()))
  return batches

def plot_with_labels(final_embeddings, filename):
  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
  plot_only = 500
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [reverse_dictionary[i] for i in range(plot_only)]

  assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
  plt.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
  plt.savefig(filename)

def check_analogy(self):
    similarities = self._sess.run([self.discriminator.similarity], feed_dict={self.train_words:self.model_params.test_items, self.dropout:1})[0]
    text = ""
    for i in range(len(self.model_params.test_items)):
        valid_word = self.model_params.reverse_dictionary[self.model_params.test_items[i][0]]
        top_k = 10  # number of nearest neighbors
        nearest = (-similarities[i, :]).argsort()[1:top_k + 1]
        log_str = 'Nearest to %s:' % valid_word
        for k in range(top_k):
            close_word = self.model_params.reverse_dictionary[nearest[k]]
            log_str = '%s %s,' % (log_str, close_word)
        text += log_str+"\n"
        with open("nlp/"+self.model_params.save_params[1]+"_analogies.txt", "w") as text_file:
          text_file.write(text)
          
    
