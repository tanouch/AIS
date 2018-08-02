import numpy as np
import os
import matplotlib.pyplot as plt
from gif_visualisation import *
from matplotlib.ticker import MultipleLocator
#matplotlib.style.use('ggplot')
#params = {'legend.fontsize': 'large',
#         'axes.labelsize': 'large',
#         'axes.titlesize':'large',
#         'xtick.labelsize':'large',
#         'ytick.labelsize':'large'}
#plt.rcParams.update(params)

#list_of_files = from_path_load_all_images("Results_StructuredPred/Results_real/CNN/")
list_of_files = list()
list_of_files = from_path_load_all_images("Results_StructuredPred/New_real/1neg_samp/")
sort_nicely(list_of_files)

uniform = np.array([elem for elem in list_of_files if "base" in elem])
selfplay = np.array([elem for elem in list_of_files if "self" in elem])
maligan = np.array([elem for elem in list_of_files if "maligan" in elem])
gan_ais = np.array([elem for elem in list_of_files if "ais" in elem])
softmax = np.array([elem for elem in list_of_files if "softmax" in elem])

uniform1 = np.array([elem for elem in uniform if "CNN" not in elem])
selfplay1 = np.array([elem for elem in selfplay if "CNN" not in elem])
maligan1 = np.array([elem for elem in maligan if "CNN" not in elem])
gan_ais1 = np.array([elem for elem in gan_ais if "CNN" not in elem])
softmax1 = np.array([elem for elem in softmax if "CNN" not in elem])

uniform2 = np.array([elem for elem in uniform if "CNN" in elem])
selfplay2 = np.array([elem for elem in selfplay if "CNN" in elem])
maligan2 = np.array([elem for elem in maligan if "CNN" in elem])
gan_ais2 = np.array([elem for elem in gan_ais if "CNN" in elem])
softmax2 = np.array([elem for elem in softmax if "CNN" in elem])

data_w2v = [uniform1, selfplay1, gan_ais1, maligan1, softmax1]
data_cnn = [uniform2, selfplay2, gan_ais2, maligan2, softmax2]
print(uniform1)
print(np.load(data_w2v[2][1]))


def moving_average(data_set, periods=3):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')

def subplot_image(raw_data, Nid, number, labels, manual_ylims, limit):
    data = [np.load(raw_data[0][Nid])[number], np.load(raw_data[1][Nid])[number], np.load(raw_data[2][Nid])[number+6], \
        np.load(raw_data[2][Nid])[number], np.load(raw_data[3][Nid])[number+6], np.load(raw_data[3][Nid])[number], np.load(raw_data[4][Nid])[number]]
    conf_int = [np.load(raw_data[0][Nid])[number+1], np.load(raw_data[1][Nid])[number+1], np.load(raw_data[2][Nid])[number+1+6], \
        np.load(raw_data[2][Nid])[number+1], np.load(raw_data[3][Nid])[number+1+6], np.load(raw_data[3][Nid])[number+1], np.load(raw_data[4][Nid])[number+1]]
    data, conf_int = [moving_average(elem) for elem in data], [moving_average(elem) for elem in conf_int]
    
    iterations = [np.arange(0, len(data[0])*1000, 1000), np.arange(0, len(data[2])*500, 500), np.arange(0, len(data[6])*1000, 1000)]
    iterations[0], data[0], data[1] = (iterations[0], data[0], data[1]) if (Nid != 0) else (iterations[0][:8], data[0][:8], data[1][:8])
    
    plt.plot(iterations[0], data[0], "g-", label="Uniform")
    plt.fill_between(iterations[0], data[0]+conf_int[0]/2, data[0]-conf_int[0]/2, color="g", alpha=0.1)

    plt.plot(iterations[0], data[1], "y-", label="Selfplay")
    plt.fill_between(iterations[0], data[1]+conf_int[1]/2, data[1]-conf_int[1]/2, color="y", alpha=0.1)

    plt.plot(iterations[1], data[2], "r-", label="AIS_gen")
    plt.fill_between(iterations[1], data[2]+conf_int[2]/2, data[2]-conf_int[2]/2, color="r", alpha=0.1)
    
    #plt.plot(iterations[1], data[3], "r--", label="AIS_disc")
    
    plt.plot(iterations[1], data[4], "b-", label="MALIGAN_gen")
    plt.fill_between(iterations[1], data[4]+conf_int[4]/2, data[4]-conf_int[4]/2, color="b", alpha=0.1)

    #plt.plot(iterations[1], data[5], "b--", label="MALIGAN_Disc")
    
    plt.plot(iterations[2], data[6], "c-", label="Softmax")
    plt.fill_between(iterations[2], data[6]+conf_int[6]/2, data[6]-conf_int[6]/2, color="c", alpha=0.1)

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    if manual_ylims:
        plt.ylim(ymin=limit[0], ymax=limit[1])
    plt.title(labels[2])
    plt.legend(loc=4)

def get_titles(Nid):
    switcher = {
        1: ["Number of Iterations", "MPR", "Amazon Dataset 1NS"], 
        2: ["Number of Iterations", "MPR", "Belgian Dataset 1NS"], 
        3: ["Number of Iterations", "MPR", "UK Dataset 1NS"], 
        4: ["Number of Iterations", "MPR", "Movielens Dataset 1NS"], 
        5: ["Number of Iterations", "MPR", "Netflix Dataset 1NS"], 
        6: ["Number of Iterations", "MPR", "Amazon Dataset 25NS"], 
        7: ["Number of Iterations", "MPR", "Belgian Dataset 25NS"], 
        8: ["Number of Iterations", "MPR", "UK Dataset 25NS"], 
        9: ["Number of Iterations", "MPR", "Movielens Dataset 25NS"], 
    }
    return switcher.get(Nid, "invalid")

def plot_graph_one_comparison(data, number, ylabel_title, manual_ylims, ylims_list):
    plt.figure(figsize=(25, 18))

    disposition = 150
    for elem in range(5):
        disposition += 1
        plt.subplot(disposition)
        labels = get_titles(elem+1)
        subplot_image(data, elem, number, labels, manual_ylims, ylims_list[elem])

    plt.savefig("results_Selfplay_vs_Tensorflow_TextCNN.pdf")


def plot_graph_one_comparison_synthe(data, number, ylabel_title, manual_ylims, ylims_list):
    plt.figure(figsize=(25, 18))

    plt.subplot(131)
    #plt.plot(iterations, np.load(data[0][0])[number][:19], "g-", label="Uniform")
    #plt.plot(iterations, np.load(data[1][0])[number][:19], "y-", label="Selfplay")
    plt.plot(iterations, np.load(data[3][0])[(number*2)][:19], "b--", label="MALIGAN_Gen")
    plt.plot(iterations, np.load(data[3][0])[(number*2)+1][:19], "b-", label="MALIGAN_Disc")
    plt.plot(iterations, np.load(data[2][0])[(number*2)][:19], "r-", label="AIS_Gen")
    plt.plot(iterations, np.load(data[2][0])[(number*2)+1][:19], "r--", label="AIS_Disc")
    plt.xlabel("Number of Iterations")
    plt.ylabel(ylabel_title)
    if manual_ylims:
        plt.ylim(ymin=ylims_list[0])
    #plt.title("Prec@1 for the Amazon dataset")
    plt.title("Blobs Dataset")
    plt.legend(loc=4)

    plt.savefig("results_Selfplay_vs_Tensorflow_TextCNN.pdf")


plot_graph_one_comparison(data_cnn, number=0, ylabel_title="MPR", manual_ylims=True, 
    ylims_list=[(76,81), (88,91.5), (88,93), (96.5,98.25), (94,96.75)]) 


#def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None, marker='d'):
#    ax = ax if ax is not None else plt.gca()
#    if np.isscalar(yerr) or len(yerr) == len(y):
#        ymin = y - yerr
#        ymax = y + yerr
#    elif len(yerr) == 2:
#        ymin, ymax = yerr
#    ax.plot(x, y, color=color, label=lab, marker=marker)
#    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)
#    return ax
#
#x = [1, 2.5, 5, 7.5, 10, 15]
#cause = [0.6987, 0.6991, 0.6998, 0.7019, 0.7022, 0.703]
#cause_STD = [0.0113, 0.0112, 0.0112, 0.0108, 0.0109, 0.0109]
#
#SP2V = [0.6693, 0.6687, 0.6724, 0.6758, 0.6784, 0.6808]
#SP2V_STD = [0.0098, 0.0098, 0.0101, 0.0102, 0.0099, 0.0101]
#
#WSP2V = [0.6853, 0.6843, 0.6844, 0.6857, 0.6883, 0.6893]
#WSP2V_STD = [0.0089, 0.0088, 0.0088, 0.0088, 0.008, 0.0088]
#
#majorLocator = MultipleLocator(1)
#
#lab = 'WSP2V-blend'
#ax = errorfill(x, np.asarray(WSP2V), np.asarray(WSP2V_STD), color='blue', marker='d')
#lab = 'SP2V-blend'
#ax = errorfill(x, np.asarray(SP2V), np.asarray(SP2V_STD), color='green', marker='+')
#lab = 'CausE-prod-C'
#ax = errorfill(x, np.asarray(cause), np.asarray(cause_STD), color='red', marker='o')
#ax.set_xlabel('Size of Test Sample in Training Set (% of Overall Dataset)')
#ax.set_ylabel('AUC Score')
#ax.xaxis.set_major_locator(majorLocator)
#ax.legend(loc='upper left')
#plt.tight_layout()
#ax.xaxis.labelpad = 10
#plt.show()