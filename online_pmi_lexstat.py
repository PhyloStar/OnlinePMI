from collections import defaultdict
import itertools as it
import sys, distances, igraph, utils
import numpy as np
import random, codecs, CRP, argparse
import DistanceMeasures as DM
from sklearn import metrics
from lingpy import *
import subprocess, clust_algos

random.seed(1234)

##TODO: Add a ML based estimation of distance or a JC model for distance between two sequences
##Separate clustering code.
##Add doculect distance as regularization


tolerance = 0.001

calpha = 0.01
pmi_weight, lexstat_weight = 3.0, 1.0

char_list = []

parser = argparse.ArgumentParser()
parser.add_argument("-mi","--max_iter", type= int, help="maximum number of iterations", default=10)
parser.add_argument("-t", "--thd", type= float, help="A number between 0 and 1 for clustering", default=0.5)
parser.add_argument("-m","--mb", type= int, help="Minibatch size", default=256)
parser.add_argument("-a","--alpha", type= float, help="alpha", default=0.75)
parser.add_argument("-M", "--margin", type= float, help="margin for filtering non-cognates", default=0.0)
parser.add_argument("-G","--gop", type= float, help="gap opening penalty", default=-2.5)
parser.add_argument("-g","--gep", type= float, help="gap extension penalty", default=-1.75)
parser.add_argument("-ca","--calpha", type= float, help="CRP alpha", default=0.1)
parser.add_argument("-i","--infile", type= str, help="input file name")
parser.add_argument("-o","--outfile", type= str, help="output file name", default="temp")
parser.add_argument("-w","--wlfile", type= str, help="input word list file name", default=None)
parser.add_argument("-A","--in_alphabet", type= str, help="input alphabet", default="asjp")
parser.add_argument("-c","--clust_algo", type= str, help="clustering algorithm name: infomap, labelprop, crp", default="infomap")
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

pmi_weight = pmi_weight/(pmi_weight+ lexstat_weight)

def lexstat_concept_evaluate_scores(d, lodict, gop, gep, tune_threshold=0.5):
    #fout = open("output.txt","w")
    average_fscore = []
    f_scores = []#defaultdict(list)
    bin_mat, n_clusters = None, 0
    for concept in d:
        ldn_dist_dict = defaultdict(lambda: defaultdict(float))
        langs = list(d[concept].keys())
        if len(langs) == 1:
            print(concept)
            continue
        scores, cognates = [], []

        for l1, l2 in it.combinations(langs, r=2):
            if d[concept][l1].startswith("-") or d[concept][l2].startswith("-"): continue
            w1, w2 = d[concept][l1], d[concept][l2]
            x, lang1 = l1
            y, lang2 = l2
            raw_score = distances.needleman_wunsch(w1, w2, scores=lodict[lang1,lang2], gop=gop, gep=gep)[0]
            if args.clust_algo == "crp":
                score = max(0.0, raw_score)
                #score = 1.0/(1.0+np.exp(-raw_score))
            else:
                score = 1.0 - (1.0/(1.0+np.exp(-raw_score)))
                #s1 = distances.needleman_wunsch(w1, w1, scores=lodict[lang1,lang1], gop=gop, gep=gep)[0]
                #s2 = distances.needleman_wunsch(w2, w2, scores=lodict[lang2,lang2], gop=gop, gep=gep)[0]
                #score = 1.0- (raw_score/((s1+s2)/2.0))
            ldn_dist_dict[l1][l2] = score
            ldn_dist_dict[l2][l1] = ldn_dist_dict[l1][l2]
        distMat = np.array([[ldn_dist_dict[ka][kb] for kb in langs] for ka in langs])

        if args.clust_algo == "crp":
            clust = CRP.gibbsCRP(distMat, crp_alpha=args.calpha, sample=False)            
        else:
            clust = clust_algos.igraph_clustering(distMat, tune_threshold, method=args.clust_algo)
        
        predicted_labels = defaultdict()
        predicted_labels_words = defaultdict()
        for k, v in clust.items():
            predicted_labels[langs[k]] = v
            predicted_labels_words[langs[k],d[concept][langs[k]]] = v
        
        predl, truel = [], []
        for l in langs:
            truel.append(cogid_dict[concept][l])
            predl.append(predicted_labels[l])
        scores = DM.b_cubed(truel, predl)
        

        #print(concept, len(langs), *scores, len(set(clust.values())), len(set(truel)),sep="\t")
        f_scores.append(list(scores))
        n_clusters += len(set(clust.values()))

    f_scores = np.mean(np.array(f_scores), axis=0)
    print("F-scores ", tune_threshold, np.round(f_scores[0],3), np.round(f_scores[1],3), np.round(f_scores[2],3), np.round(2.0*f_scores[0]*f_scores[1]/(f_scores[0]+f_scores[1]),3),sep="\t")
    return bin_mat


data_dict, cogid_dict, words_dict, langs_list, concepts_list, char_list = utils.read_data_ielex_type(dataname, reverse=args.reverse, in_alphabet=args.in_alphabet)
print("Processing ", dataname)
#print("Character list \n\n", char_list)
#print("Length of character list ", len(char_list))
#print("Language list ", langs_list)
word_list = []

#print(sys.argv)
subprocess.run(["python3", "online_pmi.py"]+sys.argv[1:])

pmidict = utils.read_pmidict(args.outfile+".pmi")

lexstat_scores = defaultdict(lambda: defaultdict(float))
denom_scores = defaultdict(lambda: defaultdict(float))

cache_scores = defaultdict()

print("Calculating numerator scores")
for l1, l2 in it.combinations_with_replacement(langs_list, r=2):# can optimize by operating on a set of words
    #print(l1, l2)
    for concept in concepts_list:
        if concept not in words_dict[l1] or concept not in words_dict[l2]:
            continue
        else:
            for w1, w2 in it.product(words_dict[l1][concept], words_dict[l2][concept]):
                algn = distances.needleman_wunsch(w1, w2, scores={}, gop=GOP, gep=GEP)[1]
                #if (w1,w2) in cache_scores:
                #    algn = cache_scores[w1,w2]
                #else:
                #    algn = distances.needleman_wunsch(w1, w2, scores={}, gop=GOP, gep=GEP)[1]
                #    cache_scores[w1,w2] = algn                            
                #    cache_scores[w2,w1] = algn
                for x, y in algn:
                    if x == "-" or y == "-":
                        continue
                    else:
                        lexstat_scores[l1,l2][x,y] += 1.0
                        lexstat_scores[l1,l2][y,x] += 1.0

print("Calculating denominator scores")
for l1, l2 in it.combinations_with_replacement(langs_list, r=2):
    #shuffle_list = concepts_list[:]
    cache_scores = defaultdict()
    print(l1, l2)
    for i in range(1000):
        #print("Iteration ",i)
        sl1, sl2 = concepts_list[:], concepts_list[:]
        random.shuffle(sl1), random.shuffle(sl2)
        for c1, c2 in zip(sl1, sl2):
            if c1 not in words_dict[l1] or c2 not in words_dict[l2]:
                continue
            else:
                for w1, w2 in it.product(words_dict[l1][c1], words_dict[l2][c2]):
                    if (w1,w2) in cache_scores:
                        algn = cache_scores[w1,w2]
                    else:
                        algn = distances.needleman_wunsch(w1, w2, scores=pmidict, gop=GEP, gep=GEP)[1]
                        cache_scores[w1,w2] = algn
                        cache_scores[w2,w1] = algn
                    
                    for x, y in algn:
                        if x == "-" or y == "-":
                            continue
                        else:
                            denom_scores[l1,l2][x,y] += 1.0
                            denom_scores[l1,l2][y,x] += 1.0

print("Finished calculation of LexStat dictionaries")
cache_scores = None

for l1, l2 in it.combinations_with_replacement(langs_list, r=2):
    #print(l1, l2)
    for x, y in it.product(char_list, char_list):
        lang_score = 0.0
        a_norm = sum(lexstat_scores[l1,l2].values())
        e_norm = sum(denom_scores[l1,l2].values())
        if (x,y) in lexstat_scores[l1,l2] and (x,y) in denom_scores:
            a_xy = lexstat_scores[l1,l2][x,y]/a_norm
            e_xy = denom_scores[l1,l2][x,y]/e_norm
            #lang_score = 2.0*(np.log(lexstat_scores[l1,l2][x,y])-np.log(denom_scores[l1,l2][x,y]))
            lang_score = 2.0*np.log(a_xy/e_xy)
        temp_score = ((1.0-pmi_weight)*lang_score)+ (pmi_weight*pmidict[x,y])
        lexstat_scores[l1,l2][x,y] = temp_score
        lexstat_scores[l2,l1][x,y] = temp_score
        #print(l1, l2, x, y, temp_score, sep="\n")

print("\nPMI scores\n")
#infomap_concept_evaluate_scores(data_dict, pmidict, GOP, GEP)

print("\nLexStat evaluation scores\n")
if args.clust_algo == "crp":
    lexstat_concept_evaluate_scores(data_dict, lexstat_scores, GOP, GEP)
else:
    for th in np.arange(0,1.0,0.05):
        bin_mat = lexstat_concept_evaluate_scores(data_dict, lexstat_scores, GOP, GEP, tune_threshold=th)

#lexstat_concept_evaluate_scores(data_dict, lexstat_scores, GOP, GEP)
    
