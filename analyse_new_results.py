import numpy as np
import os
import matplotlib.pyplot as plt
from gif_visualisation import *
from matplotlib.ticker import MultipleLocator

list_of_files = from_path_load_all_images("Results_StructuredPred/Results_real3/AIS_Selfplay/")
sort_nicely(list_of_files)
print(np.array(list_of_files))
#baseline = [np.load(elem) for elem in np.array([elem for elem in list_of_files if "base" in elem])]
#uniform = [np.load(elem) for elem in np.array([elem for elem in list_of_files if "uniform" in elem])]
selfplay = [np.load(elem) for elem in np.array([elem for elem in list_of_files if "selfplay" in elem])]
ais = [np.load(elem) for elem in np.array([elem for elem in list_of_files if "AIS" in elem])]

#
iterations = np.arange(0, 400000, 500)
plt.figure(figsize=(25, 18))
plt.subplot(141)
plt.plot(iterations[:len(selfplay[0][0])], selfplay[0][0], "g-", label="selfplay1")
plt.plot(iterations[:len(selfplay[1][0])], selfplay[1][0], "g--", label="selfplay5")
plt.plot(iterations[:len(selfplay[2][0])], selfplay[2][0], "g+", label="selfplay25")
plt.plot(iterations[:len(selfplay[3][0])], selfplay[3][0], "g*", label="selfplay50")

plt.plot(iterations[:len(ais[0][6])], ais[0][6], "y-", label="ais1")
plt.plot(iterations[:len(ais[1][6])], ais[1][6], "y--", label="ais5")
plt.plot(iterations[:len(ais[2][6])], ais[2][6], "y+", label="ais25")
plt.plot(iterations[:len(ais[3][6])], ais[3][6], "y*", label="ais50")
#plt.plot(iterations[:len(softmaxonly[0][0])], softmaxonly[0][0], "c", label="softmax")
#plt.plot(iterations[:len(softmaxsp[0][0])], softmaxsp[0][0], "c--", label="softmaxSP")
plt.legend(loc=4)
plt.ylim(ymin=89, ymax=95)

plt.subplot(142)
plt.plot(iterations[:len(selfplay[4][0])], selfplay[4][0], "g-", label="selfplay1")
plt.plot(iterations[:len(selfplay[5][0])], selfplay[5][0], "g--", label="selfplay5")
plt.plot(iterations[:len(selfplay[6][0])], selfplay[6][0], "g+", label="selfplay25")
plt.plot(iterations[:len(selfplay[7][0])], selfplay[7][0], "g*", label="selfplay50")

plt.plot(iterations[:len(ais[4][6])], ais[4][6], "y-", label="ais1")
plt.plot(iterations[:len(ais[5][6])], ais[5][6], "y--", label="ais5")
plt.plot(iterations[:len(ais[6][6])], ais[6][6], "y+", label="ais25")
plt.plot(iterations[:len(ais[7][6])], ais[7][6], "y*", label="ais50")
plt.legend(loc=4)
plt.ylim(ymin=85, ymax=95)

plt.subplot(143)
plt.plot(iterations[:len(selfplay[8][0])], selfplay[8][0], "g-", label="selfplay1")
plt.plot(iterations[:len(selfplay[9][0])], selfplay[9][0], "g--", label="selfplay5")
plt.plot(iterations[:len(selfplay[10][0])], selfplay[10][0], "g+", label="selfplay25")
plt.plot(iterations[:len(selfplay[11][0])], selfplay[11][0], "g*", label="selfplay50")

plt.plot(iterations[:len(ais[8][6])], ais[8][6], "y-", label="ais1")
plt.plot(iterations[:len(ais[9][6])], ais[9][6], "y--", label="ais5")
plt.plot(iterations[:len(ais[10][6])], ais[10][6], "y+", label="ais25")
plt.plot(iterations[:len(ais[11][6])], ais[11][6], "y*", label="ais50")
plt.legend(loc=4)
plt.ylim(ymin=96, ymax=98.5)

plt.subplot(144)
plt.plot(iterations[:len(selfplay[12][0])], selfplay[12][0], "g-", label="selfplay1")
plt.plot(iterations[:len(selfplay[13][0])], selfplay[13][0], "g--", label="selfplay5")
plt.plot(iterations[:len(selfplay[14][0])], selfplay[14][0], "g+", label="selfplay25")
plt.plot(iterations[:len(selfplay[15][0])], selfplay[15][0], "g*", label="selfplay50")

plt.plot(iterations[:len(ais[12][6])], ais[12][6], "y-", label="ais1")
plt.plot(iterations[:len(ais[13][6])], ais[13][6], "y--", label="ais5")
plt.plot(iterations[:len(ais[14][6])], ais[14][6], "y+", label="ais25")
plt.plot(iterations[:len(ais[15][6])], ais[15][6], "y*", label="ais50")
plt.legend(loc=4)
plt.ylim(ymin=93, ymax=97)

plt.savefig("fig.pdf")



def process_data():
    list_of_files = list()
    list_of_files = from_path_load_all_images("Results_StructuredPred/Results_real3/Amazon/")
    list_of_files += from_path_load_all_images("Results_StructuredPred/Results_real3/UK/")
    list_of_files += from_path_load_all_images("Results_StructuredPred/Results_real3/Belgian/")
    list_of_files += from_path_load_all_images("Results_StructuredPred/Results_real3/movielens/")
    list_of_files += from_path_load_all_images("Results_StructuredPred/Results_real3/netflix/")
    sort_nicely(list_of_files)

    baseline = np.array([elem for elem in list_of_files if "base" in elem])
    uniform = np.array([elem for elem in list_of_files if "uniform" in elem])
    selfplay = np.array([elem for elem in list_of_files if "selfplay" in elem])
    maligan = np.array([elem for elem in list_of_files if "MALIGAN" in elem])
    gan_ais = np.array([elem for elem in list_of_files if "AIS" in elem])
    softmax = np.array([elem for elem in list_of_files if "softmaxonly" in elem])
    softmax_selfplay = np.array([elem for elem in list_of_files if "softmax_sp" in elem])
    data = [baseline, uniform, selfplay, maligan, gan_ais, softmax]

    data_w2v = [np.array([elem for elem in sub_list if "CNN" not in elem]) for sub_list in data]
    data_cnn = [np.array([elem for elem in sub_list if "CNN" in elem]) for sub_list in data]

    data_w2v1 = [np.array([elem for elem in sub_list if "25" not in elem]) for sub_list in data_w2v]
    data_cnn1 = [np.array([elem for elem in sub_list if "25" not in elem]) for sub_list in data_cnn]
    data_w2v25 = [np.array([elem for elem in sub_list if "1" not in elem]) for sub_list in data_w2v]
    data_cnn25 = [np.array([elem for elem in sub_list if "1" not in elem]) for sub_list in data_cnn]

    data_w2v = [data_w2v1, data_w2v25]
    data_cnn = [data_cnn1, data_cnn25]
    return data_w2v, data_cnn


def moving_average(data_set, periods=3):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')
    #return np.array(data_set) 

def subplot_image(ax, raw_data, Nid, number, labels, manual_ylims, limit):
    data = [np.load(raw_data[0][Nid])[number], np.load(raw_data[1][Nid])[number], np.load(raw_data[2][Nid])[number], np.load(raw_data[3][Nid])[number+6], \
        np.load(raw_data[3][Nid])[number], np.load(raw_data[4][Nid])[number+6], np.load(raw_data[4][Nid])[number], np.load(raw_data[5][Nid])[number]]
    conf_int = [np.load(raw_data[0][Nid])[number+1], np.load(raw_data[1][Nid])[number+1], np.load(raw_data[2][Nid])[number+1], np.load(raw_data[3][Nid])[number+1+6], \
        np.load(raw_data[3][Nid])[number+1], np.load(raw_data[4][Nid])[number+1+6], np.load(raw_data[4][Nid])[number+1], np.load(raw_data[5][Nid])[number+1]]
    data, conf_int = [moving_average(elem) for elem in data], [moving_average(elem) for elem in conf_int]
    iterations = np.arange(0, 200000, 500)
    
    ax.plot(iterations[:len(data[0])], data[0], "g-", label="Baseline")
    ax.fill_between(iterations[:len(data[0])], data[0]+conf_int[0]/2, data[0]-conf_int[0]/2, color="g", alpha=0.1)

    ax.plot(iterations[:len(data[1])], data[1], "y-", label="Uniform")
    ax.fill_between(iterations[:len(data[1])], data[1]+conf_int[1]/2, data[1]-conf_int[1]/2, color="y", alpha=0.1)

    ax.plot(iterations[:len(data[2])], data[2], "r-", label="Selfplay")
    ax.fill_between(iterations[:len(data[2])], data[2]+conf_int[2]/2, data[2]-conf_int[2]/2, color="m", alpha=0.1)
    
    ax.plot(iterations[:len(data[3])], data[3], "k-", label="MALIGAN_gen")
    ax.fill_between(iterations[:len(data[3])], data[3]+conf_int[3]/2, data[3]-conf_int[3]/2, color="k", alpha=0.1)
    
    ax.plot(iterations[:len(data[5])], data[5], "b-", label="AIS_gen")
    ax.fill_between(iterations[:len(data[5])], data[5]+conf_int[5]/2, data[5]-conf_int[5]/2, color="b", alpha=0.1)

    ax.plot(iterations[:len(data[7])], data[7], "c-", label="Softmax")
    ax.fill_between(iterations[:len(data[7])], data[7]+conf_int[7]/2, data[7]-conf_int[7]/2, color="c", alpha=0.1)

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
        10: ["Number of Iterations", "MPR", "Netflix Dataset 25NS"], 
    }
    return switcher.get(Nid, "invalid")

def plot_graph_one_comparison(data, number, ylabel_title, manual_ylims, ylims_list):
    fig = plt.figure(figsize=(25, 18))

    disposition = 250
    for elem in range(5):
        disposition += 1
        ax = fig.add_subplot(disposition)
        labels = get_titles(elem+1)
        subplot_image(ax, data[0], elem, number, labels, manual_ylims, ylims_list[elem])
    for elem in range(4):
        disposition += 1
        ax = fig.add_subplot(disposition)
        labels = get_titles(elem+5+1)
        subplot_image(ax, data[1], elem, number, labels, manual_ylims, ylims_list[elem])

    ax = fig.add_subplot(2, 5, 10)
    labels = get_titles(10)
    subplot_image(ax, data[1], 4, number, labels, manual_ylims, ylims_list[4])

    fig.savefig("results_Selfplay_vs_Tensorflow_TextCNN.pdf")


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


def get_results_tables_aux(raw_data, number, results):
    for Nid in range(5):
        data = [np.load(raw_data[0][Nid])[number], np.load(raw_data[1][Nid])[number], np.load(raw_data[2][Nid])[number], np.load(raw_data[3][Nid])[number+6], \
            np.load(raw_data[3][Nid])[number], np.load(raw_data[4][Nid])[number+6], np.load(raw_data[4][Nid])[number], np.load(raw_data[5][Nid])[number]]
        conf_int = [np.load(raw_data[0][Nid])[number+1], np.load(raw_data[1][Nid])[number+1], np.load(raw_data[2][Nid])[number+1], np.load(raw_data[3][Nid])[number+1+6], \
            np.load(raw_data[3][Nid])[number+1], np.load(raw_data[4][Nid])[number+1+6], np.load(raw_data[4][Nid])[number+1], np.load(raw_data[5][Nid])[number+1]]
        results.append([(max(data[i])-(conf_int[i][-1]/2), max(data[i])+(conf_int[i][-1]/2)) for i in range(len(data))])
    return results

def get_results_tables(data, number=0):
    results1NS_w2v, results25NS_w2v, results1NS_cnn, results25NS_cnn = list(), list(), list(), list()
    results1NS_w2v = np.array(get_results_tables_aux(data[0][0], number, results1NS_w2v))
    results25NS_w2v = np.array(get_results_tables_aux(data[0][1], number, results25NS_w2v))
    results1NS_cnn = np.array(get_results_tables_aux(data[1][0], number, results1NS_cnn))
    results25NS_cnn = np.array(get_results_tables_aux(data[1][1], number, results25NS_cnn))
    results = [np.row_stack((results1NS_w2v[i], results25NS_w2v[i], results1NS_cnn[i], results25NS_cnn[i])) for i in range(5)]
    mean_results = np.array([[np.mean(elem) for elem in big_elem] for big_elem in results])
    print(np.array(results))
    best_results = [get_solution_name(np.argmax(elem)) for elem in results]
    print(best_results)

def get_solution_name(number):
    name = ""
    if number<16:
        name += "w2v_"
    else:
        name += "cnn"
    if number%16 < 8:
        name += "1NS_"
    else:
        name += "25NS_"
    if number%8==0:
        name += "base"
    elif number%8==1:
        name += "unif"
    elif number%8==2:
        name += "self"
    elif number%8==3:
        name += "maligan_gen"
    elif number%8==4:
        name += "maligan_disc"
    elif number%8==5:
        name += "ais_gen"
    elif number%8==6:
        name += "ais_disc"
    else:
        name += "softmax"
    return name

data_w2v, data_cnn = process_data()
plot_graph_one_comparison(data_w2v, number=0, ylabel_title="MPR", manual_ylims=True, 
    ylims_list=[(76,81), (89,95), (88,93), (96.5,98.5), (94,97)])

#get_results_tables([data_w2v, data_cnn])