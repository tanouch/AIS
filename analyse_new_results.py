import numpy as np
import math
import os
import csv
import matplotlib.pyplot as plt
from gif_visualisation import *
from matplotlib.ticker import MultipleLocator

def divide_ais_files(path):
    list_of_files = os.listdir(path)

def get_all_files_in_a_path_recursively(path):
    filenames = []
    for root, d_names, f_names in os.walk(path):
        filenames += [root+"/"+elem for elem in f_names]
    return filenames

def get_filenames_per_dataset(datasets, list_of_files):
    list_of_files_bis = {}
    for dataset in datasets:
        list_of_files_bis[dataset] = np.array([elem for elem in list_of_files if dataset in elem])
    return list_of_files_bis


def get_list_without_nan(l):
    listt = [e for e in l if not math.isnan(e)]
    if listt==[]:
        listt = [0.0]
    return listt

def screen_the_metric(list_of_files, metric_index):
    return [elem[metric_index] for elem in list_of_files]
def get_max(list_of_results):
    return [np.amax(np.array(get_list_without_nan(elem))) for elem in list_of_results]
def get_argmax(list_of_results):
    return [np.argmax(np.array(get_list_without_nan(elem))) for elem in list_of_results]
def get_specific_values(list_of_results, list_of_points, divide_factor):
    return [elem[point]/divide_factor for (elem,point) in zip(list_of_results, list_of_points)]

def get_best_algo(list_of_files_per_dataset, index):
    writer = csv.writer(open('results.csv', 'w'), delimiter=",")

    for dataset in list(list_of_files_per_dataset.keys()):
        print("")
        print(dataset)    
        writer.writerow(dataset)

        for algo in ["AIS", "selfplay", "softmax", "uniform", "baseline", "MLE"]:

            files = [elem for elem in list_of_files_per_dataset[dataset] if algo in elem]
            results = [np.load(elem) for elem in files]
            mpr = screen_the_metric(results, index)

            best_points = get_max(mpr)
            best_index = get_argmax(mpr)
            best_sorted = np.argsort(best_points)
            
            best_algo = np.array(files)[best_sorted]
            best_mpr = np.array(best_points)[best_sorted]
            best_uncertaincy = get_specific_values(screen_the_metric(results, index+1), best_index, 2)
            best_uncertaincy = np.array(best_uncertaincy)[best_sorted]
            
            final_results = list(zip(*(best_algo, best_mpr, best_uncertaincy)))
            if (len(final_results)>0):
                final_results = (final_results[-1][0], str(round(100*final_results[-1][1], 5)) + " +- "+ str(100*round(100*final_results[-1][2], 5)))
                print(algo, final_results)
                writer.writerow(final_results)
            else:
                print(algo)
                
        writer.writerow(" "+"\n")
                  
    writer.writerow(" "+"\n")
    writer.writerow(" "+"\n")


path = "Results_StructuredPred/Items_items"
list_of_files = get_all_files_in_a_path_recursively(path)
sort_nicely(list_of_files)
datasets = ["Belgian", "UK", "movielens", "netflix", "text8"]
list_of_files_per_dataset = get_filenames_per_dataset(datasets, list_of_files)


get_best_algo(list_of_files_per_dataset, 0)
get_best_algo(list_of_files_per_dataset, 2)
get_best_algo(list_of_files_per_dataset, 4)


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

#data_w2v, data_cnn = process_data()
#plot_graph_one_comparison(data_w2v, number=0, ylabel_title="MPR", manual_ylims=True, 
#    ylims_list=[(76,81), (89,95), (88,93), (96.5,98.5), (94,97)])

#get_results_tables([data_w2v, data_cnn])