import math
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from tools import *

def print_results_predictions(last_predictions, batch_inputs, targets, vocabulary_size):
    predictions_for_targets = 100*np.mean(np.array([last_predictions[i][int(targets[i])] for i in range(len(batch_inputs))]))
    predictions_for_random = 100*np.mean(np.array([last_predictions[i][np.random.choice(vocabulary_size-1, 20, replace=False)] for i in range(len(batch_inputs))]))

    predictions_top_one = [1 if targets[i] in list(np.argpartition(-last_predictions[i], 1)[:1]) else 0 for i in range(len(batch_inputs))]
    predictions_top_one = 100*np.count_nonzero(np.array(predictions_top_one))/len(batch_inputs)
    predictions_top_five = [1 if targets[i] in list(np.argpartition(-last_predictions[i], 5)[:5]) else 0 for i in range(len(batch_inputs))]
    predictions_top_five = 100*np.count_nonzero(np.array(predictions_top_five))/len(batch_inputs)

    percent = int(vocabulary_size*0.01)
    predictions_top_one_percent = [1 if targets[i] in list(np.argpartition(-last_predictions[i], percent)[:percent]) else 0 for i in range(len(batch_inputs))]
    predictions_top_one_percent = 100*np.count_nonzero(np.array(predictions_top_one_percent))/len(batch_inputs)

    predictions_sorted = [1-len(np.where(last_predictions[i]>last_predictions[i][int(targets[i])])[0])/vocabulary_size for i in range(len(batch_inputs))]
    MPR_sorted = 100*np.mean(np.array(predictions_sorted))
    predictions_sorted_random = [1-(len(np.where(last_predictions[i]>last_predictions[i][int(np.random.random_integers(vocabulary_size-1))])[0])/vocabulary_size) for i in range(len(batch_inputs))]
    return predictions_top_one, predictions_top_one_percent, MPR_sorted
    #plt.figure(figsize=(15, 5))
    #plt.hist(predictions_for_random.flatten(), bins=150, range=(0,np.percentile(last_predictions[0], 95)), alpha=0.5, normed=True, label="randoms")
    #plt.hist(predictions_for_targets.flatten(), bins=150, range=(0,np.percentile(last_predictions[0], 95)), alpha=0.5, normed=True, label="targets")
    #plt.legend()
    #plt.savefig("image")

def get_conf_int(data, num_X_val):
    return norm.interval(0.95, loc=np.mean(np.array(data)), scale=np.std(np.array(data))/math.sqrt(num_X_val))

def append_results_MPR_and_prec(self, i, operation, MPRs, p1, p1p, train_words, targets):
    last_predictions = self._sess.run([operation], feed_dict={self.train_words:train_words, self.dropout:1})[0]
    precision1, precision1p, MPR = print_results_predictions(last_predictions, train_words, targets, self.vocabulary_size)
    MPRs.append(MPR)
    p1.append(precision1)
    p1p.append(precision1p)

def append_lists_with_zeros(list_of_lists):
    for elem in list_of_lists:
        elem.append(0.0)

def print_confidence_intervals(self, mode):
    all_MPRs_G, all_MPRs_D, all_precisions_1G, all_precisions_1D, all_precisions_1pG, all_precisions_1pD = list(), list(), list(), list(), list(), list()
    numbers = [len(elem) for elem in self.test_list_batches][2:]
    number_of_cross_validations = 5
    for k in range(number_of_cross_validations):
        MPR_G, MPR_D, p1G, p1D, p1pG, p1pD = list(), list(), list(), list(), list(), list()

        for i in range(2, len(self.test_list_batches)):
            if (len(self.test_list_batches[i])>2):
                batches_selected = np.random.choice(np.arange(len(self.test_list_batches[i])), int(0.7*len(self.test_list_batches[i])), replace=False)
                batches = self.test_list_batches[i][batches_selected]
                train_words, targets = batches[:,:-1], np.reshape(batches[:,-1], (-1,1))
                append_results_MPR_and_prec(self, i, self.before_softmax_D, MPR_D, p1D, p1pD, train_words, targets)
                if (mode!="selfplay") and (mode!="baseline"):
                    append_results_MPR_and_prec(self, i, self.before_softmax_G, MPR_G, p1G, p1pG, train_words, targets)
            else:
                append_lists_with_zeros([MPR_G, MPR_D, p1G, p1D, p1pG, p1pD])
        #if k==0:
        #    print(MPR_D)
        all_MPRs_D.append(np.dot(np.array(numbers), np.array(MPR_D))/np.sum(numbers))
        all_precisions_1D.append(np.dot(np.array(numbers), np.array(p1D))/np.sum(numbers))
        all_precisions_1pD.append(np.dot(np.array(numbers), np.array(p1pD))/np.sum(numbers))
        if (mode!="selfplay") and (mode!="baseline"):
            all_MPRs_G.append(np.dot(np.array(numbers), np.array(MPR_G))/np.sum(numbers))
            all_precisions_1G.append(np.dot(np.array(numbers), np.array(p1G))/np.sum(numbers))
            all_precisions_1pG.append(np.dot(np.array(numbers), np.array(p1pG))/np.sum(numbers))

    mpr, p1, p1p = get_conf_int(all_MPRs_D, 5), get_conf_int(all_precisions_1D, 5), get_conf_int(all_precisions_1pD, 5)
    print("Results for D " + "MPR "+ str(mpr)+ " P@1 "+ str(p1)+ " P@1p " + str(p1p))
    def print_confidence_intervals_aux(l, mpr, p1, p1p):
        l[0].append(round(sum(mpr)/2, 5))
        l[1].append(round(mpr[1]-mpr[0], 5))
        l[2].append(round(sum(p1)/2, 5))
        l[3].append(round(p1[1]-p1[0], 5))
        l[4].append(round(sum(p1p)/2, 5))
        l[5].append(round(p1p[1]-p1p[0], 5))

    print_confidence_intervals_aux([self.MPRD, self.confintD, self.p1D, self.confintp1D, self.p1pD, self.confintp1pD], mpr, p1, p1p)

    if (mode!="selfplay") and (mode!="baseline"):
        mpr, p1, p1p = get_conf_int(all_MPRs_G, 5), get_conf_int(all_precisions_1G, 5), get_conf_int(all_precisions_1pG, 5)
        print("Results for G " + "MPR "+ str(mpr)+ " P@1 "+ str(p1)+ " P@1p " + str(p1p))
        print_confidence_intervals_aux([self.MPRG, self.confintG, self.p1G, self.confintp1G, self.p1pG, self.confintp1pG], mpr, p1, p1p)
    print("")

#Functions AUC for Synthetic datasets !!!
def get_Z_value(Z, speeding_factor, elem):
    Z = np.transpose(Z)
    return np.mean([Z[int(elem[0]/speeding_factor), int(elem[1]/speeding_factor)], Z[int(elem[0]/speeding_factor)-1, int(elem[1]/speeding_factor)],
        Z[int(elem[0]/speeding_factor), int(elem[1]/speeding_factor)-1], Z[int(elem[0]/speeding_factor)-1, int(elem[1]/speeding_factor)-1]])

def get_proportion_of_true_negatives(Z, threshold, speeding_factor, negative_samples):
    true_negatives = [elem for elem in negative_samples if get_Z_value(Z, 25, elem) < threshold]
    false_negatives = [elem for elem in negative_samples if get_Z_value(Z, 25, elem) > threshold]
    true_neg_prop = 100*len(true_negatives)/len(negative_samples)
    return true_neg_prop, np.array(true_negatives), np.array(false_negatives)

def calculate_auc(self):
    positive_samples = self.test_data[np.random.choice(len(self.test_data), size=1000)]
    negative_samples = [(np.random.randint(self.vocabulary_size), np.random.randint(self.vocabulary_size)) for elem in range(5000)]
    _, negative_samples, _ = get_proportion_of_true_negatives(self.model_params.Z, self.model_params.threshold, \
        self.model_params.speeding_factor, negative_samples)

    positive_samples_new = (positive_samples/self.model_params.speeding_factor).astype(int)
    negative_samples_new = (negative_samples/self.model_params.speeding_factor).astype(int)
    positive_scores = [self.heatmap[elem[0], elem[1]] for elem in positive_samples_new]
    negative_scores = [self.heatmap[elem[0], elem[1]] for elem in negative_samples_new]
    scores = positive_scores+negative_scores
    labels = np.array([1]*len(positive_samples) + [0]*len(negative_samples))
    auc_score = roc_auc_score(labels, scores)
    return auc_score

def get_heatmap_density(self, positive_samples, output_distributions):
    heatmap = np.zeros((int(self.vocabulary_size/self.model_params.speeding_factor), int(self.vocabulary_size/self.model_params.speeding_factor)))
    heatmap_final = np.zeros((int(self.vocabulary_size/self.model_params.speeding_factor), int(self.vocabulary_size/self.model_params.speeding_factor)))
    for i, positive_sample in enumerate(positive_samples):
        distrib_compressed = np.mean(np.reshape(output_distributions[i], (-1, self.model_params.speeding_factor)), axis=1)
        heatmap[int(positive_sample[0]/self.model_params.speeding_factor)] += distrib_compressed
    for i in range(len(heatmap)):
        heatmap_final[i] = np.mean(heatmap[max(0, i-5):min(int(self.vocabulary_size/self.model_params.speeding_factor)-1, i+5)], axis=0)
    sum_tot = np.sum(heatmap_final)
    heatmap_final = heatmap_final/sum_tot
    self.heatmap = heatmap_final


def print_samples_and_KDE_density(self, true_negative_samples, false_negative_samples, print_params, print_negatives):
    plt.figure(figsize=(15, 5))
    if print_negatives:
        plt.plot(true_negative_samples[:,0], true_negative_samples[:,1], 'o', markersize=2, color="black", alpha=0.045, label="Negative samples")
        plt.plot(false_negative_samples[:,0], false_negative_samples[:,1], 'o', markersize=2, color="red", alpha=0.045, label="Negative samples")

    plt.imshow(np.transpose(self.heatmap), extent=[0, self.vocabulary_size, 0, self.vocabulary_size], alpha=0.4, cmap=cm.RdYlGn, origin="lower", aspect='auto', interpolation="bilinear")
    plt.contour(self.model_params.X, self.model_params.Y, self.model_params.Z, levels=[self.model_params.threshold], label="KDE estimation")
    plt.legend(loc=4)
    plt.xlabel("Input Product")
    plt.ylabel("Target Product")
    plt.title("Probability distributions of the Target product wrt the Input product for : " + print_params[0] + " " + print_params[1]+ " model")

    folder = self.model_params.folder+"/"+print_params[0]+"/"
    create_dir(folder)
    folder = folder + "/" + print_params[1] + "/"
    create_dir(folder)
    if print_negatives:
        folder = folder+"/"+"Negatives"+"/"
        create_dir(folder)
        plt.savefig(folder+print_params[1]+"_neg"+str(self.pic_number))
    else:
        folder = folder+"/"+"Normal"+"/"
        create_dir(folder)
        plt.savefig(folder+print_params[1]+str(self.pic_number))


def create_dir(type_of_data):
    if not os.path.exists(type_of_data):
        os.makedirs(type_of_data)

def print_candidate_sampling(self, print_params):
    batches = self.test_data[np.random.choice(len(self.test_data), size=2500)]
    train_words, label_words = np.reshape(batches[:,0], (-1,1)), np.reshape(batches[:,-1], (-1,1))

    if (print_params[1]=="DISC"):
        output_distributions, negative_samples= self._sess.run([self.output_distributions_D,self.disc_fake_samples], \
            feed_dict={self.train_words:train_words, self.label_words:label_words, self.dropout:1})
    if (print_params[1]=="GEN"):
        output_distributions, negative_samples= self._sess.run([self.output_distributions_G , self.gen_fake_samples], \
            feed_dict={self.train_words:train_words, self.label_words:label_words, self.dropout:1})

    negative_pairs = np.column_stack((np.tile(train_words, (1, negative_samples.shape[1])).flatten(), negative_samples.flatten()))

    true_neg_prop, true_negative_samples, false_negative_samples = get_proportion_of_true_negatives(self.model_params.Z, \
        self.model_params.threshold, 25, negative_pairs)
    get_heatmap_density(self, batches, output_distributions)

    print_samples_and_KDE_density(self, true_negative_samples, false_negative_samples, print_params, False)
    print_samples_and_KDE_density(self, true_negative_samples, false_negative_samples, print_params, True)

    auc_score = calculate_auc(self)
    self.pic_number += 1

    print("")
    print("Proportion of true negatives = " + print_params[0]+ "_" + print_params[1] +" "+ str(true_neg_prop))
    print("AUC score is "+ print_params[0]+ "_" + print_params[1] +" "+ str(auc_score))
    print("")
    return true_neg_prop, auc_score
