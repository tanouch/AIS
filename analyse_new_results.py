import numpy as np
import math
import os
import csv
import matplotlib.pyplot as plt
from gif_visualisation import *
from matplotlib.ticker import MultipleLocator

def divide_ais_files(path):
    list_of_files = os.listdir(path)

def rename_files(list_of_files):
    for files in list_of_files:
        
        if ("uniform" in files):
            os.rename(files, files.replace('_uniform', ''))
        if ("selfplay" in files):
            os.rename(files, files.replace('_selfplay', ''))
        if ("context" in files):
            os.rename(files, files.replace('_context', ''))

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
    dictionnary = {}
    for dataset in list(list_of_files_per_dataset.keys()):
        #writer.writerow(dataset)
        dictionnary[dataset] = list()
        for algo in ["AIS", "selfplay", "softmax", "uniform", "baseline"]:

            files = [elem for elem in list_of_files_per_dataset[dataset] if algo in elem]
            results = [np.load(elem) for elem in files]
            results = [elem for elem in results if len(elem[0])>1]
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
                final_results = (str(round(100*final_results[-1][1], 5)) + " +-"+ str(round(100*final_results[-1][2], 5)))
                dictionnary[dataset].append(final_results)
            else:
                dictionnary[dataset].append([])
                #writer.writerow(final_results)
        #writer.writerow(" "+"\n")                  
    #writer.writerow(" "+"\n")
    return dictionnary


path = "Results_StructuredPred/items_items/Full"
list_of_files = get_all_files_in_a_path_recursively(path)
sort_nicely(list_of_files)

#AIS = [files for files in list_of_files if "AIS" in files]
#softmax = [files for files in list_of_files if "softmax" in files]
#baseline = [files for files in list_of_files if "baseline" in files]
#MLE = [files for files in list_of_files if "MLE" in files]
#rename_files(AIS)
#rename_files(softmax)
#rename_files(MLE)
#rename_files(baseline)

#datasets = ["Belgian", "UK", "movielens", "netflix", "text8", "text9"]
#datasets = ["Belgian", "UK", "movielens", "netflix"]
datasets = ["blobs0", "blobs1", "blobs2", "swiss_roll"]
list_of_files_per_dataset = get_filenames_per_dataset(datasets, list_of_files)

dictionnary_MPR = get_best_algo(list_of_files_per_dataset, 0)
dictionnary_AUC = get_best_algo(list_of_files_per_dataset, 2)
dictionnary_prec1 = get_best_algo(list_of_files_per_dataset, 4)
#dictionnary_MPR["movielens"][1] = str((0.94911319145888514+0.9461378124803244)/2) + "+-" + str((0.94911319145888514-0.9461378124803244))
#dictionnary_prec1["movielens"][1] = str((0.023666796043040082-0.021333203956959916)/2) + "+-" + str((0.023666796043040082-0.021333203956959916))
#dictionnary_MPR["movielens"][1] = str((96.31909+96.22)/2) + " +-" + str((96.319-96.226))
#dictionnary_prec1["movielens"][1] = str((3.002+3.32)/2) + " +-" + str((3.325-3.0025))
#dictionnary_MPR["Belgian"][1] = str((89.1569572+89.27)/2) + " +-" + str((89.277-89.156))
#dictionnary_prec1["Belgian"][1] = str((10.68954+10.886)/2) + " +-" + str((10.886-10.689))

#UK AIS mpr (0.8699239549971418, 0.8704703454700242) prec1 (0.0277474385911871(0.964844423465945, 0.96591206372695) prec1 (0.024283334761877415, 0.02775666523812258), 0.030372561408812893)
#Belgian AIS mpr (0.8939068294988197, 0.8955689021344648) prec1 (0.10585003206315695, 0.10938996793684307)
#Text8 AIS mpr (0.8723160047708505, 0.8744992222935746) prec1 (0.07446470877929368, 0.07753529122070632)
#Text9 AIS mpr (0.8935673961159635, 0.896812785877957) prec1 (0.061166529009579716, 0.06499347099042029)
#Netflix mpr (0.9477666443059527, 0.9504909088735487) prec1 (0.020202749579806778, 0.021637250420193224)
#Movielens mpr (0.9632064085091614, 0.9645566373176506) prec1 (0.03368729176995822, 0.03539270823004178)

for dataset in list(list_of_files_per_dataset.keys()):
    print("")
    print(dataset)
    for i, algo in enumerate(["AIS", "selfplay", "softmax", "uniform", "baseline"]):
        print(algo, dictionnary_MPR[dataset][i], dictionnary_AUC[dataset][i], dictionnary_prec1[dataset][i])


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