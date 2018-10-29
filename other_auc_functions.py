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

def get_proportion_of_true_negatives_new(self, pairs):
    positive_pairs, negative_pairs = list(), list()
    summ = 0
    for elem in pairs:
        if (elem[0] not in self.model_params.dict_co_occurences):
            summ +=1
            negative_pairs.append(elem)
        else:
            if (self.model_params.dict_co_occurences[elem[0]][elem[1]]==0):
                negative_pairs.append(elem)
            else:
                positive_pairs.append(elem)
    print(len(positive_pairs), summ)
    return np.array(positive_pairs), np.array(negative_pairs)
