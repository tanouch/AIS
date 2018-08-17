import math
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import norm, describe
from sklearn.metrics import roc_auc_score

from tools import *
from training_utils import create_batch

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

def print_confidence_intervals(self):
    all_MPRs_G, all_MPRs_D, all_precisions_1G, all_precisions_1D, all_precisions_1pG, all_precisions_1pD = list(), list(), list(), list(), list(), list()
    numbers = [len(elem) for elem in self.test_list_batches][2:]
    number_of_cross_validations = 6
    for k in range(number_of_cross_validations):
        MPR_G, MPR_D, p1G, p1D, p1pG, p1pD = list(), list(), list(), list(), list(), list()

        for i in range(2, len(self.test_list_batches)):
            if (len(self.test_list_batches[i])>2):
                batches_selected = np.random.choice(np.arange(len(self.test_list_batches[i])), int(0.7*len(self.test_list_batches[i])), replace=False)
                batches = self.test_list_batches[i][batches_selected]
                train_words, targets = batches[:,:-1], np.reshape(batches[:,-1], (-1,1))
                append_results_MPR_and_prec(self, i, self.before_softmax_D, MPR_D, p1D, p1pD, train_words, targets)
                if (self.model_params.model_type=="AIS") or (self.model_params.model_type=="MALIGAN"):
                    append_results_MPR_and_prec(self, i, self.before_softmax_G, MPR_G, p1G, p1pG, train_words, targets)
            else:
                append_lists_with_zeros([MPR_G, MPR_D, p1G, p1D, p1pG, p1pD])
        
        all_MPRs_D.append(np.dot(np.array(numbers), np.array(MPR_D))/np.sum(numbers))
        all_precisions_1D.append(np.dot(np.array(numbers), np.array(p1D))/np.sum(numbers))
        all_precisions_1pD.append(np.dot(np.array(numbers), np.array(p1pD))/np.sum(numbers))
        if (self.model_params.model_type=="AIS") or (self.model_params.model_type=="MALIGAN"):
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

    print_confidence_intervals_aux([self.main_scoreD, self.confintD, self.p1D, self.confintp1D, self.p1pD, self.confintp1pD], mpr, p1, p1p)

    if (self.model_params.model_type=="AIS") or (self.model_params.model_type=="MALIGAN"):
        mpr, p1, p1p = get_conf_int(all_MPRs_G, 5), get_conf_int(all_precisions_1G, 5), get_conf_int(all_precisions_1pG, 5)
        print("Results for G " + "MPR "+ str(mpr)+ " P@1 "+ str(p1)+ " P@1p " + str(p1p))
        print_confidence_intervals_aux([self.main_scoreG, self.confintG, self.p1G, self.confintp1G, self.p1pG, self.confintp1pG], mpr, p1, p1p)
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

def get_auc_synthetic_data(self, print_params):
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

def get_auc_real_data(self):
    positive_samples = self.test_data[np.random.choice(len(self.test_data), size=5000)]
    negative_samples = np.array([(np.random.randint(self.vocabulary_size2), np.random.randint(self.vocabulary_size2)) for elem in range(5000)])
    
    number_of_cross_validations = 7
    auc_scoresD, auc_scoresG = list(), list()
    for k in range(number_of_cross_validations):
        batches_selected = np.random.choice(np.arange(len(positive_samples)), int(0.7*len(positive_samples)), replace=False)
        def append_res(op, auc_scores):
            pos_t, pos_l = np.reshape(positive_samples[batches_selected][:,0], (-1,1)), np.reshape(positive_samples[batches_selected][:,1], (-1,1))
            neg_t, neg_l = np.reshape(negative_samples[batches_selected][:,0], (-1,1)), np.reshape(negative_samples[batches_selected][:,1], (-1,1))
            positive_scores = list(self._sess.run([op], feed_dict={self.train_words:pos_t, self.label_words:pos_l, self.dropout:1})[0])
            negative_scores = list(self._sess.run([op], feed_dict={self.train_words:neg_t, self.label_words:neg_l, self.dropout:1})[0])
            scores = positive_scores+negative_scores
            labels = np.array([1]*len(positive_scores) + [0]*len(negative_scores))
            auc_scores.append(roc_auc_score(labels, scores))

        append_res(self.score_target_D, auc_scoresD)
        if (self.model_params.model_type=="AIS") or (self.model_params.model_type=="MALIGAN"):
            append_res(self.score_target_G, auc_scoresG)

    print("")
    def get_auc_real_data_aux(l, auc, model):
        auc = get_conf_int(auc, 5)
        print("AUC score for " + model + str(auc))
        l[0].append(round(sum(auc)/2, 5))
        l[1].append(round(auc[1]-auc[0], 5))
        
    get_auc_real_data_aux([self.main_scoreD, self.confintD], auc_scoresD, "discriminator")
    if (self.model_params.model_type=="AIS") or (self.model_params.model_type=="MALIGAN"):
            get_auc_real_data_aux([self.main_scoreG, self.confintG], auc_scoresG, "generator")
    print("")

def get_auc_and_true_neg_proportion(self, print_params):
    if (self.model_params.type_of_data=="synthetic"):
        get_auc_synthetic_data(self, print_params)
    else:
        get_auc_real_data(self, print_params)

def get_proportion_same_element(self):
    train_words, label_words = create_batch(self, 1000)
    if (self.model_params.model_type == "AIS") or (self.model_params.model_type == "MALIGAN"):
        samples = self._sess.run(self.gen_self_samples, feed_dict={self.train_words:train_words, self.label_words:label_words, self.dropout:1})
        tv, fv = self._sess.run([self.true_values_sigmoid, self.fake_values_sigmoid], feed_dict={self.train_words:train_words, self.label_words:label_words, self.dropout:1})
        auc = roc_auc_score(np.array([1]*len(tv.flatten())+[0]*len(fv.flatten())), np.array(list(tv.flatten())+list(fv.flatten())))
        acc = [np.mean(np.around(tv)), 1-np.mean(np.around(fv))]
        print("auc disc ", str(auc))
        print('acc disc true and fake ', str(acc))
    else:
        samples = self._sess.run(self.disc_self_samples, feed_dict={self.train_words:train_words, self.label_words:label_words, self.dropout:1})

    prop, prop_equal_target = list(), list()
    for i, elem in enumerate(samples):
        prop.append(len(np.unique(elem))/len(elem))
        prop_equal_target.append(len(np.where(elem==label_words[:,-1][i])[0])/len(elem))

    print("Proportion unique " + str(round(100*np.mean(np.array(prop)), 5)))
    print("Proportion equal target " + str(round(100*np.mean(np.array(prop_equal_target)), 5)))
    self.true_neg_prop.append(np.mean(np.array(prop)))

