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
        print("Step", str(step))
        print_gen_and_disc_losses(self, step)

        if (self.model_params.type_of_data=="real"):
            calculate_mpr(self)
            calculate_auc_discriminator(self, step)
            check_similarity(self, step)
            check_analogies(self, step)
            #calculate_proportion_same_element(self, step)
       
        if (self.model_params.type_of_data=="synthetic"):
            synthetic_data_auc_and_plotting(self)
        print("")

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


def calculate_mpr(self):
    calculate_mpr_and_auc_pairs(self, self.before_softmax_D, self.dataD, "Disc")
    if ("AIS" in self.model_params.model_type):
        calculate_mpr_and_auc_pairs(self, self.before_softmax_G, self.dataG, "Gen")


def mpr_func(self, batches, context_words, label_words, distributions):
    if (len(batches[0])==2):
        mpr, prec1 = 0, 1
        for k in range(len(batches)):
            if (self.model_params.dataset in ["blobs0", "blobs1", "blobs50", "blobs100", "blobs200", "blobs500", "swiss_roll", "s_curve", "moons"]):
                l = list_elem_not_co_occuring(self, context_words[k][0], label_words[k][0])
                rank = rankdata(distributions[k][l])[-1]
                prec1 = prec1+1 if (rank>(len(l)-2)) else prec1
                mpr += rank/len(l)
            else: 
                rank = rankdata(distributions[k])[label_words[k][0]]
                prec1 = prec1+1 if (rank>(self.model_params.vocabulary_size2-2)) else prec1
                mpr += rank/self.model_params.vocabulary_size2
        mpr /= len(batches)
        prec1 /= len(batches)
    
    else:
        mpr = np.mean(np.array([rankdata(distributions[k])[label_words[k]]/self.model_params.vocabulary_size \
                for k in range(len(batches))]))
        prec1 = 1
    return mpr, prec1


def calculate_mpr_and_auc_pairs(self, op, list_of_logs, net):
    mprs, aucs, prec1s = list(), list(), list()
    batch = self.test_data[:2000] if (self.model_params.dataset in ["blobs50", "blobs100", "blobs200", "swiss_roll", "s_curve", "moons"]) \
        else self.test_data[:7000]

    for i in range(5):
        batches = batch[np.random.choice(len(batch), size=1000)] if (self.model_params.dataset in ["blobs0", "blobs1", "blobs2", "swiss_roll", "s_curve", "moons"]) \
            else batch[np.random.choice(len(batch), size=5000)]
        context_words, label_words = np.reshape(batches[:,0], (-1,1)), np.reshape(batches[:,-1], (-1,1))

        if self.model_params.use_pretrained_embeddings:
            user_embeddings = self.model_params.user_embeddings[batches[:,0]]
            distributions = np.matmul(user_embeddings, np.transpose(self.model_params.item_embeddings))
        else:
            distributions = self._sess.run(op, feed_dict={self.train_words:context_words, self.dropout:1})
        
        positive_scores = [distributions[i][label_words[i][0]]*self.model_params.true_popularity_distributions[context_words[i][0]] for i in range(len(batches))]
        negative_scores = [distributions[i][e]*self.model_params.true_popularity_distributions[context_words[i][0]] for i in range(len(batches)) for e in get_negatives(self, context_words[i][0], 10)] 
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


def synthetic_data_auc_and_plotting(self):
    #batches = self.test_data[np.random.choice(len(self.test_data), size=5000)]
    batches = self.test_data[:5000]
    context_words, label_words = np.reshape(batches[:,0], (-1,1)), np.reshape(batches[:,-1], (-1,1))
    feed_dict = {self.train_words:context_words, self.label_words:label_words, self.dropout:1}

    if (self.model_params.model_type=="baseline") or (self.model_params.model_type=="softmax") or (self.model_params.model_type=="MLE"):
        out_dist = self._sess.run(self.output_distributions_D, feed_dict)
        ns = [1]
    else:
        out_dist, ns= self._sess.run([self.output_distributions_D, self.disc_self_samples], feed_dict)
        false_ns = sum([1 for context in context_words.flatten() for neg in ns[context] if self.model_params.Z[neg,context]<self.model_params.threshold])/len(ns.flatten())

    print_samples_and_KDE_density(self, batches, ns, [1], out_dist, "DISC")
    mpr, auc = calculate_mpr_and_auc_pairs(self, self.output_distributions_D, self.dataD, "D")

    if (self.model_params.model_type=="AIS") or (self.model_params.model_type=="MALIGAN"):
        out_dist, ns, ns_D_values = self._sess.run([self.output_distributions_G , self.gen_self_samples, self.gen_self_values_D_sigmoid_aux], feed_dict)
        print_samples_and_KDE_density(self, batches, ns, ns_D_values, out_dist, "GEN")
        mpr, auc = calculate_mpr_and_auc_pairs(self, self.output_distributions_G, self.dataG, "G")
    
    print("Number of different targets sampled", len(np.unique(ns.flatten())))
    print("Proportion of true negative sampled", false_ns)
    print("")


def get_heatmap_density(self, positive_samples, output_distributions):
    feed_dict = {self.train_words:np.reshape(np.arange(self.model_params.vocabulary_size), (-1,1)), self.dropout:1}
    heatmap = self._sess.run(self.output_distributions_G, feed_dict) if self.model_params.model_type=="AIS" else self._sess.run(self.output_distributions_D, feed_dict)
    return heatmap

def print_samples_and_KDE_density(self, batches, negative_samples, d_values, output_distributions, net_type):
    context_words, label_words = np.reshape(batches[:,0], (-1,1)), np.reshape(batches[:,-1], (-1,1))
    heatmap = get_heatmap_density(self, batches, output_distributions)
    #positive_scores = [heatmap[i,j] for (i,j) in self.test_data[:2000]]
    #random = np.reshape(np.random.choice(self.model_params.vocabulary_size2, 5000),(-1,2))
    #negative_scores = [heatmap[i,j] for (i,j) in random if self.model_params.Z[j,i]<self.model_params.threshold]
    #labels, logits = np.array([1]*len(positive_scores) + [0]*len(negative_scores)), np.array(positive_scores+negative_scores)
    #print(roc_auc_score(labels, logits))
    
    def plot_samples(print_negatives):
        plt.figure(figsize=(20, 12))
        if print_negatives:
            negative_pairs = np.column_stack((np.tile(context_words, (1, negative_samples.shape[1])).flatten(), negative_samples.flatten()))
            plt.plot(negative_pairs[:,0], negative_pairs[:,1], 'o', markersize=2, color="black", alpha=0.08, label="Negative samples")
            #plt.scatter(negative_pairs[:,0], negative_pairs[:,1], marker='o', s=2, c=d_values.flatten(), alpha=0.08)

        #plt.imshow(heatmap, extent=[0, self.vocabulary_size, 0, self.vocabulary_size], alpha=0.4, cmap=cm.RdYlGn, origin="lower", aspect='auto', interpolation="bilinear")
        binary_data = np.zeros((self.vocabulary_size2, self.vocabulary_size2))
        X, Y = np.meshgrid(np.arange(self.model_params.vocabulary_size2), np.arange(self.model_params.vocabulary_size2))
        for elem in self.model_params.data:
            binary_data[elem[0], elem[1]] = 1
        plt.contour(Y, X, binary_data, levels=[0.9], label="KDE estimation")
        plt.legend(loc=4)
        plt.xlabel("Input Product")
        plt.ylabel("Target Product")
        plt.title("Probability distributions of the Target product wrt the Input product for : " + self.model_params.model_type + " " + net_type+ " model")

        if (self.model_params.model_type != "SS"):
            folder = self.model_params.folder+"/"+self.model_params.model_type+"/"
        else:
            folder = self.model_params.folder+"/"+self.model_params.model_type+"_"+"_".join(self.model_params.sampling)+"/"
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
            op = self.disc_self_samples
    
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
            self.gen_self_values_D_sigmoid, self.gen_self_values_D_softmax,
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
    if ("AIS" in self.model_params.model_type) and (step%1000==0):
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
    if ("text" in self.model_params.dataset) or (self.model_params.dataset == "UK"):
        writer = csv.writer(open(self.model_params.name+"_"+str(step)+'.csv', 'w'), delimiter=",")
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
                context_embeddings.append(embeddings[elem[0]] - embeddings[elem[1]] + embeddings[elem[2]])
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