import math
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import norm, describe, rankdata
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from tools import *
from training_utils import create_batch, print_gen_and_disc_losses, early_stopping

def testing_step(self, step):
    if (step % self.model_params.printing_step == 0):
        print_gen_and_disc_losses(self, step)

    if (step % self.model_params.printing_step == 0):
        print("Step", str(step))
        if (self.model_params.type_of_data=="real"):
            calculate_mpr(self)
            calculate_auc_discriminator(self, step)
            #calculate_proportion_same_element(self, step)
            check_similarity(self, step)
            check_analogies(self, step)
       
        if (self.model_params.type_of_data=="synthetic"):
            get_auc_synthetic_data(self)

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

def calculate_mpr_baskets(self):
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
                if ("AIS" in self.model_params.model_type) or (self.model_params.model_type=="MALIGAN"):
                    append_results_MPR_and_prec(self, i, self.before_softmax_G, MPR_G, p1G, p1pG, train_words, targets)
            else:
                append_lists_with_zeros([MPR_G, MPR_D, p1G, p1D, p1pG, p1pD])
        
        all_MPRs_D.append(np.dot(np.array(numbers), np.array(MPR_D))/np.sum(numbers))
        all_precisions_1D.append(np.dot(np.array(numbers), np.array(p1D))/np.sum(numbers))
        all_precisions_1pD.append(np.dot(np.array(numbers), np.array(p1pD))/np.sum(numbers))
        if ("AIS" in self.model_params.model_type) or (self.model_params.model_type=="MALIGAN"):
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

    if ("AIS" in self.model_params.model_type) or (self.model_params.model_type=="MALIGAN"):
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


def calculate_mpr(self):
    if self.model_params.working_with_pairs:
        calculate_mpr_and_auc_pairs(self, self.output_distributions_D, self.dataD, "Disc")
        if ("AIS" in self.model_params.model_type):
            calculate_mpr_and_auc_pairs(self, self.output_distributions_G, self.dataG, "Gen")
    else:    
        calculate_mpr_baskets(self)

def mpr_func(self, batches, context_words, label_words, distributions):
    if (len(batches[0])==2):
        mpr, prec1 = 0, 1
        for k in range(len(batches)):
            l = list_elem_not_co_occuring(self, context_words[k][0], label_words[k][0])
            rank = rankdata(distributions[k][l])[-1]
            prec1 = prec1+1 if (rank>(len(l)-2)) else prec1
            mpr += rank/len(l)
        mpr /= len(batches)
        prec1 /= len(batches)
    
    else:
        mpr = np.mean(np.array([rankdata(distributions[k])[label_words[k]]/self.model_params.vocabulary_size \
                for k in range(len(batches))]))
        prec1 = 1
    return mpr, prec1


def calculate_mpr_and_auc_pairs(self, op, list_of_logs, net):
    mprs, aucs, prec1s = list(), list(), list()
    batch = self.test_data[:10000]
    #test_context = np.random.choice(np.unique(self.test_data[:,0]), 1000) #batch = [[e, np.random.choice(self.model_params.dict_co_occurences[e])]]

    for i in range(5):
        batches = batch[np.random.choice(len(batch), size=7000)]
        context_words, label_words = np.reshape(batches[:,0], (-1,1)), np.reshape(batches[:,-1], (-1,1))

        if self.model_params.use_pretrained_embeddings:
            user_embeddings = self.model_params.user_embeddings[batches[:,0]]
            distributions = np.matmul(user_embeddings, np.transpose(self.model_params.item_embeddings))
            
        else:
            distributions = self._sess.run(op, \
                feed_dict={self.train_words:context_words, self.label_words:label_words, self.dropout:1})
        
        positive_scores = [distributions[i][label_words[i][0]]*self.model_params.popularity_distributions[label_words[i][0]] for i in range(len(batches))]
        negative_scores = [distributions[i][e]*self.model_params.popularity_distributions[label_words[i][0]] for i in range(len(batches)) for e in get_negatives(self, context_words[i][0], 10)] \
        #if (len(batches[0])==2) else [distributions[i][e] for e in np.random.choice(self.model_params.vocabulary_size, 10) for i in range(len(batches))]
        
        labels, logits = np.array([1]*len(positive_scores) + [0]*len(negative_scores)), np.array(positive_scores+negative_scores)
        aucs.append(roc_auc_score(labels, logits))
        
        mpr, prec1 = mpr_func(self, batches, context_words, label_words, distributions)
        mprs.append(mpr)
        prec1s.append(prec1)
    
    mpr = get_conf_int(mprs, 5)
    auc = get_conf_int(aucs, 5)
    prec1 = get_conf_int(prec1s, 5)
    print(net, "auc", auc, "mpr", mpr, "prec1", prec1)
    res = [mpr, auc, prec1]
    for i in range(3):
        list_of_logs[2*i].append((res[i][0]+res[i][1])/2)
        list_of_logs[2*i + 1].append(res[i][1]-res[i][0])
    return mpr, auc

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
    return heatmap_final

def print_samples_and_KDE_density(self, batches, negative_samples, d_values, output_distributions, net_type):
    context_words, label_words = np.reshape(batches[:,0], (-1,1)), np.reshape(batches[:,-1], (-1,1))
    heatmap = get_heatmap_density(self, batches, output_distributions)
    
    def plot_samples(print_negatives):
        plt.figure(figsize=(20, 12))
        if print_negatives:
            negative_pairs = np.column_stack((np.tile(context_words, (1, negative_samples.shape[1])).flatten(), negative_samples.flatten()))
            true_neg_prop, true_negative_pairs, false_negative_pairs = get_proportion_of_true_negatives(self.model_params.Z, self.model_params.threshold, 25, negative_pairs)
            if len(d_values)==1:
                plt.plot(negative_pairs[:,0], negative_pairs[:,1], 'o', markersize=2, color="black", alpha=0.08, label="Negative samples")
            else:
                plt.scatter(negative_pairs[:,0], negative_pairs[:,1], marker='o', s=2, c=d_values.flatten(), alpha=0.08)

        plt.imshow(np.transpose(heatmap), extent=[0, self.vocabulary_size, 0, self.vocabulary_size], alpha=0.4, cmap=cm.RdYlGn, origin="lower", aspect='auto', interpolation="bilinear")
        plt.contour(self.model_params.X, self.model_params.Y, self.model_params.Z, levels=[self.model_params.threshold], label="KDE estimation")
        plt.legend(loc=4)
        plt.xlabel("Input Product")
        plt.ylabel("Target Product")
        plt.title("Probability distributions of the Target product wrt the Input product for : " + self.model_params.model_type + " " + net_type+ " model")

        if (self.model_params.model_type != "SS"):
            folder = self.model_params.folder+"/"+self.model_params.model_type+"/"
        else:
            folder = self.model_params.folder+"/"+self.model_params.model_type+"_"+self.model_params.discriminator_samples_type+"/"
        create_dir(folder)
        folder = folder + "/" + net_type + "/"
        create_dir(folder)
        
        folder = folder+"/"+"Negatives"+"/" if print_negatives else folder+"/"+"Normal"+"/"
        create_dir(folder)
        plt.savefig(folder+net_type+"_neg"+str(self.pic_number)+".pdf")
        plt.savefig(folder+net_type+"_neg"+str(self.pic_number))
        self.pic_number += 1

    plot_samples(False)
    if (len(negative_samples)!=1):
        plot_samples(True)


def get_auc_synthetic_data(self):
    print("")
    batches = self.test_data[np.random.choice(len(self.test_data), size=5000)]
    context_words, label_words = np.reshape(batches[:,0], (-1,1)), np.reshape(batches[:,-1], (-1,1))

    if (self.model_params.model_type=="baseline") or (self.model_params.model_type=="softmax") or (self.model_params.model_type=="MLE"):
        output_distributions = self._sess.run(self.output_distributions_D, \
        feed_dict={self.train_words:context_words, self.dropout:1})
        negative_samples = [1]
    else:
        output_distributions, negative_samples= self._sess.run([self.output_distributions_D, self.disc_random_and_self_samples], \
        feed_dict={self.train_words:context_words, self.dropout:1})

    print_samples_and_KDE_density(self, batches, negative_samples, [1], output_distributions, "DISC")
    mpr, auc = calculate_mpr_and_auc_pairs(self, self.output_distributions_D, self.dataD, "D")

    if (self.model_params.model_type=="AIS") or (self.model_params.model_type=="MALIGAN"):
        output_distributions, negative_samples, negative_samples_D_values = self._sess.run([self.output_distributions_G , self.gen_self_samples, self.gen_self_values_D], \
            feed_dict={self.train_words:context_words, self.label_words:label_words, self.dropout:1})
        print_samples_and_KDE_density(self, batches, negative_samples, negative_samples_D_values, output_distributions, "GEN")
        mpr, auc = calculate_mpr_and_auc_pairs(self, self.output_distributions_G, self.dataG, "G")
    print("")


def get_auc_real_data(self):
    positive_samples = self.test_data[np.random.choice(len(self.test_data), size=5000)]
    negative_samples = np.array([(np.random.randint(self.vocabulary_size2), np.random.randint(self.vocabulary_size2)) for elem in range(5000)])
    
    number_of_cross_validations = 5
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


def calculate_proportion_same_element(self, step):
    if (self.model_params.neg_sampled>1) and (self.model_params.model_type != "baseline") and \
        (self.model_params.model_type != "softmax") and (self.model_params.model_type != "MLE"):
        if self.model_params.working_with_pairs :
            batches = self.training_data[np.random.choice(len(self.training_data), 500)]
        else:
            batches = get_basket_set_size_training(self.training_data, 500, 3)

        train_words, label_words = batches[:,:-1], batches[:,1:]
        feed_dict={self.train_words:train_words, self.label_words:label_words, self.dropout:1}
        if ("AIS" in self.model_params.model_type) or (self.model_params.model_type == "MALIGAN"):    
            op = self.gen_self_samples 
        elif (self.model_params.model_type == "SS") or (self.model_params.model_type == "BCE"):
            op = self.disc_random_and_self_samples
    
        samples = self._sess.run(op, feed_dict)
        prop, prop_equal_target = list(), list()
        for i, elem in enumerate(samples):
            prop.append(len(np.unique(elem))/len(elem))
            prop_equal_target.append(len(np.where(elem==batches[i][-1])[0])/len(elem))
        print("Proportion unique " + str(100*np.mean(np.array(prop))))
        print("Proportion equal target " + str(100*np.mean(np.array(prop_equal_target))))
        print("")
        if ("AIS" in self.model_params.model_type) or (self.model_params.model_type == "MALIGAN"):
            self.dataG[-1].append(np.mean(np.array(prop)))
        else:
            self.dataD[-1].append(np.mean(np.array(prop)))
        del samples, op

def get_training_auc_discriminator_aux(self, batches):
    train_words, label_words = batches[:,:-1], batches[:,1:]
    fake_sig, fake_soft, true_sig, true_soft = self._sess.run([
            self.gen_fake_values_D_sigmoid, self.gen_fake_values_D_softmax,
            self.gen_true_values_D_sigmoid, self.gen_true_values_D_softmax], 
        feed_dict={self.train_words:train_words, self.label_words:label_words, self.dropout:1})
    
    def print_rankings(fake_disc, true_disc):
        rankings = list()
        for i in range(len(batches)):
            ranking = rankdata(list(fake_disc[i])+[true_disc[i]])[-1]
            rankings.append(ranking)
        average_ranking = (len(fake_disc[0])+1 - sum(rankings)/len(batches))/(len(fake_disc[0])+1)
        print("Average true rank disc", round(average_ranking, 2))
    
    def print_auc(fake_disc, true_disc, title):
        scores = np.concatenate((fake_disc.flatten(), true_disc.flatten()))
        labels = np.array([0]*len(fake_disc.flatten())+[1]*len(true_disc.flatten()))
        auc = roc_auc_score(labels, scores)
        print(title, auc)
    print_auc(fake_sig, true_sig, "auc sigmoid")
    print_auc(fake_soft, true_soft, "auc softmax")
    
def calculate_auc_discriminator(self, step):
    if ("AIS" in self.model_params.model_type) and (step%500==0):
        if self.model_params.working_with_pairs:
            batches = self.training_data[np.random.choice(len(self.training_data), 1000)]
            get_training_auc_discriminator_aux(self, batches)
        else:
            for seq_length in [2, 4, 6]:
                print(seq_length)
                batches = get_basket_set_size_training(self.training_data, 200, seq_length)
                get_training_auc_discriminator_aux(self, batches)
        print("")

def check_similarity(self, step):
    writer = csv.writer(open(self.model_params.name+"_"+str(step)+'.csv', 'w'), delimiter=",")
    
    if ("text" in self.model_params.dataset) or (self.model_params.dataset == "UK"):
        if self.model_params.use_pretrained_embeddings:
            norm = np.sqrt(np.sum(np.square(self.model_params.item_embeddings), 1, keepdims=True))
            normalized_embeddings = self.model_params.item_embeddings/norm
            item_embeddings = normalized_embeddings[self.model_params.check_words_similarity]
            similarities = np.matmul(item_embeddings, np.transpose(normalized_embeddings))
        else:
            feed_dict={self.train_words:np.reshape(self.model_params.check_words_similarity, (-1,1)), self.dropout:1}
            op = self.similarity_D if ("AIS" not in self.model_params.model_type) else self.similarity_G
            similarities = self._sess.run(op, feed_dict)
        
        closest_words = list()
        for i, scores in enumerate(similarities):
            ids = np.argsort(-scores)[:25]
            closest = [self.model_params.dictionnary[Id] for Id in ids]
            closest_words.append(closest)
            writer.writerow((closest[0], closest[1:]))
            #print(self.model_params.dictionnary[self.model_params.check_words_similarity[i]], closest[0], closest[1:])

def check_analogies(self, step):
    if ("text" in self.model_params.dataset):
        def acc_list_analogies(self, list_analogies):
            context_embeddings = list()
            if self.model_params.use_pretrained_embeddings:
                embeddings = self.model_params.item_embeddings
            else:
                op = self.discriminator.embeddings_tensorflow if ("AIS" not in self.model_params.model_type) else self.generator.embeddings_tensorflow
                embeddings = self._sess.run(op)
            
            for elem in list_analogies:
                context_embeddings.append(embeddings[elem[0]] - \
                    embeddings[elem[1]] + embeddings[elem[2]])
            context_embeddings = normalize(np.array(context_embeddings), norm='l2', axis=1, copy=True, return_norm=False)
            embeddings = normalize(embeddings, norm='l2', axis=1, copy=True, return_norm=False)
            analogies = np.matmul(context_embeddings, np.transpose(embeddings))

            acc2, acc5, acc15 = 0, 0, 0
            for i, scores in enumerate(analogies):
                ids = np.argsort(-scores)
                if (list_analogies[i][-1] in ids[:2]):
                    acc2 += 1
                if (list_analogies[i][-1] in ids[:5]):
                    acc5 += 1
                if (list_analogies[i][-1] in ids[:15]):
                    acc15 += 1
            del embeddings
            return round(100*acc2/len(list_analogies), 3), round(100*acc5/len(list_analogies), 3), round(100*acc15/len(list_analogies), 3)

        sem2, sem5, sem15 = acc_list_analogies(self, self.model_params.list_semantic_analogies)
        syn2, syn5, syn15 = acc_list_analogies(self, self.model_params.list_syntactic_analogies)
        print("Sem analogy acc", sem2, sem5, sem15, "Syn analogy acc", syn2, syn5, syn15)