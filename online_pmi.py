from collections import defaultdict
import itertools as it
import sys, distances, igraph, json
from utils import *
import numpy as np
import random, codecs
import DistanceMeasures as DM
#from sklearn import metrics
#from lingpy import *
import argparse
import CRP, clust_algos
random.seed(1234)

##TODO: Add a ML based estimation of distance or a JC model for distance between two sequences
##Separate clustering code.
##Add doculect distance as regularization
#number of Iterations, threshold, minbatch, alpha, GOP, GEP, temp file, margin
#read cognate_class column, data column by header
#outputfile with cognate judgments, no evaluation
#print pmi matrix, add option of initial cognate list
#Add list of hyperparameters as suffix
#Thank Gerhard for labelprop idea
#Add option to directly read PMI matrix
#Fix IPA, Add CRP

tolerance = 0.001


parser = argparse.ArgumentParser()
parser.add_argument("-mi","--max_iter", type= int, help="maximum number of iterations", default=10)
parser.add_argument("-t", "--thd", type= float, help="A number between 0 and 1 for clustering", default=0.5)
parser.add_argument("-m","--mb", type= int, help="Minibatch size", default=256)
parser.add_argument("-a","--alpha", type= float, help="alpha", default=0.75)
parser.add_argument("-M", "--margin", type= float, help="margin for filtering non-cognates", default=0.0)
parser.add_argument("-G","--gop", type= float, help="gap opening penalty", default=-2.5)
parser.add_argument("-g","--gep", type= float, help="gap extension penalty", default=-1.75)
parser.add_argument("-ca","--calpha", type= float, help="CRP alpha", default=0.01)
parser.add_argument("-i","--infile", type= str, help="input file name")
parser.add_argument("-o","--outfile", type= str, help="output file name", default="temp")
parser.add_argument("-w","--wlfile", type= str, help="input word list file name", default=None)
parser.add_argument("-A","--in_alphabet", type= str, help="input alphabet", default="asjp")
parser.add_argument("-c","--clust_algo", type= str, help="clustering algorithm name: infomap, labelprop, crp", default="labelprop")
parser.add_argument("-e","--eval", help="evaluate cognate clusters", action='store_true')
parser.add_argument("-s","--nw_th", help="threshold for NW vanilla word similarity score", type=float, default=0.0)
parser.add_argument("-S","--ldn_th", help="threshold for ldn vanilla word similarity score", type=float, default=1.0)
parser.add_argument("-I","--issim", help="select LDN or NW", type=str, default="ldn")
parser.add_argument("-N","--nexus", help="generate a nexus file", action='store_true')
parser.add_argument("-R","--reverse", help="read string reverse", action='store_true')
parser.add_argument("-p","--prune", help="prune word list", action='store_true')
parser.add_argument("--sample", help="sample crp alpha", action='store_true')
parser.add_argument("-O","--optimize", help="Optimize gap opening and gap extension penalties", action='store_true')


args = parser.parse_args()

MAX_ITER, infomap_threshold, min_batch, margin, GOP, GEP, alpha = args.max_iter, args.thd, args.mb, args.margin, args.gop, args.gep, args.alpha

dataname = args.infile
outname = args.outfile

char_list = []
 
def infomap_concept_evaluate_scores(d, lodict, gop, gep, lang_list):
    average_fscore = []
    f_scores = []
    bin_mat, n_clusters = [], 0
    f_preds = open(args.outfile+".cognates", "w")
    f_preds.write("gloss\tlanguage\t"+args.in_alphabet+"\tcognate class\n")
    for concept in d:
        ldn_dist_dict = defaultdict(lambda: defaultdict(float))
        langs = list(d[concept].keys())
        if len(langs) == 1:
            print(concept, " removed since it has only language to cluster")
            continue
        scores, cognates = [], []

        for l1, l2 in it.combinations(langs, r=2):
            if d[concept][l1].startswith("-") or d[concept][l2].startswith("-"): continue

            w1, w2 = d[concept][l1], d[concept][l2]
            raw_score = distances.needleman_wunsch(w1, w2, scores=lodict, gop=gop, gep=gep)[0]
            if args.clust_algo == "crp":
                score = max(0.0, raw_score)
                #score = 1.0/(1.0+np.exp(-raw_score))
            else:
                score = 1.0 - (1.0/(1.0+np.exp(-raw_score)))
            #print(w1, w2,raw_score, score)
            ldn_dist_dict[l1][l2] = score
            ldn_dist_dict[l2][l1] = ldn_dist_dict[l1][l2]
        distMat = np.array([[ldn_dist_dict[ka][kb] for kb in langs] for ka in langs])
        
        if args.clust_algo == "crp":
            clust = CRP.gibbsCRP(distMat, crp_alpha=args.calpha, sample=False)            
        else:
            clust = clust_algos.igraph_clustering(distMat, infomap_threshold, method=args.clust_algo)

        predicted_labels, predicted_labels_words = defaultdict(), defaultdict()

        for k, v in clust.items():
            predicted_labels[langs[k]] = v
            predicted_labels_words[langs[k],d[concept][langs[k]]] = v
            f_preds.write(concept+"\t"+ langs[k][1]+"\t"+ d[concept][langs[k]]+"\t"+ str(v)+"\n")

        predl, truel = [], []
        if args.eval:
            for l in langs:
                truel.append(cogid_dict[concept][l])
                predl.append(predicted_labels[l])
                
            scores = DM.b_cubed(truel, predl)
            #print(concept, len(langs), scores[0], scores[1], scores[2], len(set(clust.values())), len(set(truel)), sep="\t")
            f_scores.append(list(scores))
        n_clusters += len(set(clust.values()))
        if args.nexus:
            t = dict2binarynexus(predicted_labels, lang_list)
            bin_mat += t

    if args.eval:
        f_scores = np.mean(np.array(f_scores), axis=0)
        print("Fscores",f_scores[0], f_scores[1], 2.0*f_scores[0]*f_scores[1]/(f_scores[0]+f_scores[1]), sep="\t")
    f_preds.close()
    return bin_mat

def calc_pmi(alignment_dict, char_list, scores, initialize=False, init_const = 0.001):
    sound_dict = defaultdict(float)
    relative_align_freq = 0.0
    relative_sound_freq = 0.0
    count_dict = defaultdict(float)
    
    if initialize == True:
        for c1, c2 in it.product(char_list, repeat=2):
            if c1 == "-" or c2 == "-":
                continue
            count_dict[c1,c2] += init_const
            count_dict[c2,c1] += init_const
            sound_dict[c1] += init_const
            sound_dict[c2] += init_const
            relative_align_freq += init_const
            relative_sound_freq += 2.0*init_const

    for alignment, score in zip(alignment_dict, scores):
        #score = 1.0
        for a1, a2 in alignment:
            if a1 == "-" or a2 == "-":
                continue
            count_dict[a1,a2] += 1.0*score
            count_dict[a2,a1] += 1.0*score
            sound_dict[a1] += 2.0*score
            sound_dict[a2] += 2.0*score
            #relative_align_freq += 2.0
            #relative_sound_freq += 2.0

    relative_align_freq = sum(list(count_dict.values()))
    relative_sound_freq = sum(list(sound_dict.values()))
    #print(count_dict, sound_dict)
    for a in count_dict.keys():
        m = count_dict[a]
        if m <=0: print(a, m)
        assert m>0

        num = np.log(m)-np.log(relative_align_freq)
        denom = np.log(sound_dict[a[0]])+np.log(sound_dict[a[1]])-(2.0*np.log(relative_sound_freq))
        val = num - denom
        count_dict[a] = val
        #count_dict[a] = val/(-1.0*num)
    return count_dict

data_dict, cogid_dict, words_dict, langs_list, concepts_list, char_list = read_data_ielex_type(dataname, reverse=args.reverse, in_alphabet=args.in_alphabet)
print("Processing ", dataname)
print("Character list \n\n", " ".join(char_list))
print("Length of character list ", len(char_list))
print("Language list ", langs_list,sep="\n")

word_list = []

if args.wlfile is not None:
    for line in open(args.wlfile, "r"):
        x, y, s = line.split("\t")
        word_list += [[x,y]]
else:
    for concept in data_dict:
        print(concept)
        words = []
        for idx in data_dict[concept]:
            words.append(data_dict[concept][idx])
        for x, y in it.combinations(words, r=2):
            if args.issim =="ldn":
                if distances.ldn(x, y) <=args.ldn_th:
                    word_list += [[x,y]]
            else:
                if distances.needleman_wunsch(x, y, scores=None, gop=-1,gep=-1)[0] >= args.nw_th:
                    word_list += [[x,y]]


n_examples, n_updates = len(word_list), 0
n_wl = len(word_list)
print("Size of initial list ", n_wl)

pmidict = defaultdict(float)
net_sim = [0.0]*MAX_ITER

for n_iter in range(0,MAX_ITER):
    random.shuffle(word_list)
    pruned_wl = []
    n_zero = 0.0
    print("Iteration ", n_iter)
    for idx in range(0, n_wl, min_batch):
        wl = word_list[idx:idx+min_batch]
        eta = np.power(n_updates+2, -alpha)
        algn_list, scores = [], []
        #print("Processed example number ", idx)
        for w1, w2 in wl:
            #print(w1,w2,sc)
            if not pmidict:
                s, alg = distances.needleman_wunsch(w1, w2, scores={}, gop=GOP, gep=GEP)
            else:
                s, alg = distances.needleman_wunsch(w1, w2, scores=pmidict, gop=GOP, gep=GEP)
            if s <= margin:
                n_zero += 1.0
                if args.prune:
                    continue
            net_sim[n_iter-1] += max(0,s)

            algn_list.append(alg)
            scores.append(distances.sigmoid(s))

            if args.prune:
                pruned_wl.append([w1, w2])
        mb_pmi_dict = calc_pmi(algn_list, char_list, scores, initialize=True)
        for k, v in mb_pmi_dict.items():
            pmidict_val = pmidict[k]
            pmidict[k] = (eta*v) + ((1.0-eta)*pmidict_val)
        n_updates += 1
    print("Size of word list ", n_wl)
    print("Non zero examples ", n_wl-n_zero)
    print("Number of updates ", n_updates)
    print("Net similarity ", net_sim[n_iter-1])
    
    #infomap_concept_evaluate_scores(data_dict, pmidict, GOP, GEP, langs_list)
    #GOP, GEP = optimize_gop_gep(pmidict, words_dict, langs_list, concepts_list)
    if n_iter%10 == 0 and args.optimize:
        GOP, GEP = scipy_optimize_gop_gep(pmidict, words_dict, langs_list, concepts_list, GOP, GEP)
        print("Optimized parameters ", GOP, GEP)
    #bin_mat = infomap_concept_evaluate_scores(data_dict, pmidict, GOP, GEP, langs_list)
    if args.prune:
        word_list = pruned_wl[:]
    n_wl = len(word_list)
    #infomap_concept_evaluate_scores(data_dict, pmidict, -2.5, -1.75)


print("\nWriting PMI scores\n")
pmi_fw = open(args.outfile+".pmi", "w")
for k, v in pmidict.items():
    print(k[0], k[1], v, sep="\t", file=pmi_fw)
pmi_fw.close()

bin_mat = infomap_concept_evaluate_scores(data_dict, pmidict, GOP, GEP, langs_list)

if args.nexus:
    nchar, nlangs = np.array(bin_mat).shape

    fw = open(outname+".nex","w")
    fw.write("begin data;"+"\n")
    fw.write("   dimensions ntax="+str(nlangs)+" nchar="+str(nchar)+";\nformat datatype=restriction interleave=no missing= ? gap=-;\nmatrix\n")

    for row, lang in zip(np.array(bin_mat).T, langs_list):
        rowx = "".join([str(x) for x in row])
        fw.write(lang+"\t"+rowx.replace("2","?")+"\n")
    fw.write(";\nend;")

