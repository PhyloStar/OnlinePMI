import itertools as it
import numpy as np
from collections import defaultdict

def gibbsCRP(pair_dist, crp_alpha=0.01, max_iter=5):
    """A gibbs sampling based CRP. Does nothing much but moves the items around.
    """

    n_items = pair_dist.shape[0]
    assert n_items > 0
    
    #Initialize cluster list
    cluster_idx = [[x] for x in range(n_items)]
    n_iter = 1
    clusters_vec, cluster_idxs = [], []
    cluster_idxs.append([x for x in cluster_idx])
    
    ##For the first iteration. Initialize things.
    self_similarity_vec = []
    
    temp_ss_vec = []
    for c in cluster_idx:
        c_sum = 0.0
        for i, j in it.product(c, c):
            c_sum += pair_dist[i][j]
        temp_ss_vec.append(c_sum)
    
    self_similarity_vec.append(temp_ss_vec[:])
    
    bcubed_prec, bcubed_recall, bcubed_fscore = [], [], []
    ari_vec = []
    previous_cluster_idx = [list(x) for x in cluster_idxs[-1]]
    previous_self_similarity_vec =  self_similarity_vec[-1]
    n_gibbs_step = 0.0
    ari, f_score = 0.0, 0.0
    #print("\tMEANING ", gloss)
    while (n_iter <= max_iter):
        #random.shuffle(items_list)
        #print("\tIteration ", n_iter)
        for item in range(n_items):#Find the maximum similar cluster
            n_gibbs_step += 1.0
            cluster_idx = [list(x) for x in previous_cluster_idx]#make a copy of the last cluster idxs. Change it
            for j, j_item in enumerate(cluster_idx):
                if item in j_item:
                    temp_index = j_item.index(item)
                    del cluster_idx[j][temp_index]
                  
            cluster_idx = [x for x in cluster_idx if x != []]##remove empty clusters
            cluster_sum_vec = []#temporary similarity function
            
            if cluster_idx == []:
                cluster_idx.append([item])
                continue
            cluster_sum_vec.append(crp_alpha)
            for j_cluster in cluster_idx:
                cluster_sum = 0.0
                for k_j_item in j_cluster:
                    p = pair_dist[item][k_j_item]
                    cluster_sum += p
                cluster_sum_vec.append(cluster_sum)
            #print(cluster_sum_vec)
            cluster_sum_vec = np.array(cluster_sum_vec)/np.sum(cluster_sum_vec)
            
            #insert_index = np.random.choice(range(len(cluster_sum_vec)),p=cluster_sum_vec)
            insert_index = np.argmax(cluster_sum_vec)
            #cluster_idx[insert_index].append(item)
            if insert_index == 0:
                cluster_idx.append([item])
            else:
                cluster_idx[insert_index-1].append(item)
            #print("\t", n_gibbs_step, item, insert_index, cluster_idx[insert_index-1], "\n")
            previous_cluster_idx = [x for x in cluster_idx if x != []]#remove empty clusters
            
            #predicted_labels, gold_labels = [], []
            #cluster_idxs.append(previous_cluster_idx)
            
            #for k_idx, k in enumerate(previous_cluster_idx):
            #    for k_item in k:
            #        predicted_labels.append(int(k_idx))
            #        gold_labels.append(gold_dict[int(k_item.split("::")[1])])
            #p, r, f_score = b_cubed(gold_labels,predicted_labels)
            #bcubed_prec.append(float(p))
            #bcubed_recall.append(float(r))
            #bcubed_fscore.append(float(f_score))
            #ari = metrics.adjusted_rand_score(gold_labels, predicted_labels)
            #ari_vec.append(ari)
            
        #print cluster_sum_vec, len(cluster_sum_vec)
        #n_clusters = len(previous_cluster_idx)
        #f.write("\t"+str(n_gibbs_step)+"\t"+str(n)+"\t"+ str(n_clusters)+ "\t"+ str(f_score)+"\t"+str(ari)+"\n")
        #for clu in previous_cluster_idx:
        #    print("\t",clu)
        #print previous_cluster_idx, len(previous_cluster_idx)
        n_iter += 1

    #f.write("\tScores"+ "\t"+ gloss+ "\t"+ str(bcubed_fscore[-1]) + "\t"+ str(ari_vec[-1])+ "\t"+ str(len(previous_cluster_idx))+ "\t"+ str(len(set(gold_labels)))+"\n")
    #print(previous_cluster_idx)
    clust = defaultdict()
    for cluster_idx, cluster in enumerate(previous_cluster_idx):
        for idx in cluster:
            clust[idx] = cluster_idx
    return clust


