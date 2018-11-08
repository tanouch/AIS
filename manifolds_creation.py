from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.neighbors.kde import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_s_curve, make_swiss_roll, make_moons, make_blobs, make_circles
from scipy.stats import describe
import math
import os
from sklearn.metrics import roc_auc_score

def create_dir(type_of_data):
    if not os.path.exists(type_of_data):
        os.makedirs(type_of_data)

def create_true_data(type_of_data, number_of_modes, std, size, vocabulary_size):
    list_of_x_values, list_of_y_values = list(), list()
    if (type_of_data=="mixture_of_gaussians"):
        for i in range(number_of_modes):
            list_of_x_values.append(np.clip(np.random.normal(loc=np.random.randint(vocabulary_size-1), scale=500, size=size), 0, vocabulary_size))
            list_of_y_values.append(np.clip(np.random.normal(loc=np.random.randint(vocabulary_size-1), scale=500, size=size), 0, vocabulary_size))
        x = np.column_stack((np.append([], list_of_x_values), np.append([], list_of_y_values)))

    cos_theta = np.random.uniform()
    sin_theta = math.sqrt(1-cos_theta*cos_theta)
    if (type_of_data=="blobs"):
        x = np.clip(((vocabulary_size/20)*make_blobs(n_samples=size, centers=number_of_modes, cluster_std=std)[0]+(vocabulary_size/2)), [0,0], [vocabulary_size, vocabulary_size]).astype(int)
    if (type_of_data=="moons"):
        x = ((np.dot(make_moons(n_samples=size)[0]*(1/2), np.array([[cos_theta, sin_theta], [-sin_theta, cos_theta]])))*(vocabulary_size/2)+(vocabulary_size/2)).astype(int)
    if (type_of_data=="circles"):
        x = ((make_circles(n_samples=size)[0]*(vocabulary_size/2))+(vocabulary_size/2)).astype(int)
    if (type_of_data=="swiss_roll"):
        x = make_swiss_roll(n_samples=size, random_state=2, noise=std)[0]
        x = np.column_stack((x[:,0], x[:,2]))
        x = np.dot((1/25)*x,np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]]))
        x = (x*(vocabulary_size/2)+(vocabulary_size/2)).astype(int)
    if (type_of_data=="s_curve"):
        x = make_s_curve(n_samples=size)[0]/2
        x = np.column_stack((x[:,0], x[:,2]))
        x = ((np.dot(x, np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])))*(vocabulary_size/2)+(vocabulary_size/2)).astype(int)
    return x

def non_linear_transformations(x, y, vocabulary_size):
    return np.clip(((x*y)/(1+x+y)).astype(int), 0, vocabulary_size), np.clip(((x*x+y*y)/(1+x+y)).astype(int), 0, vocabulary_size)
def non_linear_transformations2(x, y, vocabulary_size):
    return x, np.clip((np.power((y-vocabulary_size/2), 2)/(y+1)).astype(int), 0, vocabulary_size)

def print_data_points(x, y):
    plt.figure(figsize=(15, 5))
    min_x, max_x, min_y, max_y, unzoom_factor = np.amin(x), np.amax(x), np.amin(y), np.amax(y), 500
    plt.xlim((min_x-unzoom_factor, max_x+unzoom_factor))
    plt.ylim((min_y-unzoom_factor, max_y+unzoom_factor))
    plt.plot(x, y, 'ro')
    #sns.kdeplot(x, y, cmap="Reds", shade=True, shade_lowest=False, cumulative=False, vertical=False, cut=3)
    plt.legend()
    plt.show()

def print_percentile_data(data):
    percentile_true_data_1 = np.percentile(data, 1)
    percentile_true_data_50 = np.percentile(data, 50)

    percentile_true_data_99 = np.percentile(data, 99)
    print(percentile_true_data_1, percentile_true_data_50, percentile_true_data_99)
    percentile_true_data = np.percentile(data, 0)
    return percentile_true_data

def print_data_points_and_KDE_density(x, KDE, type_of_data, folder):
    create_dir(folder)
    speeding_factor = 1
    X,Y = np.meshgrid(np.arange(0, vocabulary_size, speeding_factor), np.arange(0, vocabulary_size, speeding_factor))
    mesh = np.column_stack([X.ravel(), Y.ravel()])
    true_values = np.exp(KDE.score_samples(x))
    threshold = print_percentile_data(true_values)

    Z = np.exp(KDE.score_samples(mesh))
    print_percentile_data(Z)
    true_indices = np.where(Z>threshold)[0]
    true_data = np.column_stack([X.ravel()[true_indices], Y.ravel()[true_indices]])
    Z = np.reshape(Z, (int(vocabulary_size/speeding_factor), int(vocabulary_size/speeding_factor)))

    plt.figure(figsize=(15, 5))
    plt.contour(X, Y, Z, levels=[threshold])
    plt.savefig(folder+"/"+type_of_data)

    np.save(folder+"/"+"true_data.npy", true_data)
    np.save(folder+"/"+"X.npy", X)
    np.save(folder+"/"+"Y.npy", Y)
    np.save(folder+"/"+"Z.npy", Z)
    np.save(folder+"/"+"threshold.npy", threshold)

if True:
    type_of_data = "blobs"
    vocabulary_size, number_of_points_generated, number_of_modes, std = 1000, 10000, 500, 0.05
    x = create_true_data(type_of_data, number_of_modes, std, number_of_points_generated, vocabulary_size)
    print_data_points(x[:,0], x[:,1])
    KDE = KernelDensity(bandwidth=25.0).fit(x)
    print_data_points_and_KDE_density(x, KDE, type_of_data, "blobs5")
