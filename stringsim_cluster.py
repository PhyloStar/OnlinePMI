from collections import defaultdict
import itertools as it
import sys, distances, igraph, json
from utils import *
import numpy as np
import random, codecs
import DistanceMeasures as DM
#from sklearn import metrics
from lingpy import *
import argparse
random.seed(1234)

##TODO: Add a ML based estimation of distance or a JC model for distance between two sequences
##Separate clustering code.
##Add doculect distance as regularization
#number of Iterations, threshold, minbatch, alpha, GOP, GEP, temp file, margin
#read cognate_class column, data column by header
#outputfile with cognate judgments, no evaluation
#print pmi matrix, add option of initial cognate list

tolerance = 0.001

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--thd", type= float, help="A number between 0 and 1 for clustering", default=0.5)
parser.add_argument("-i","--infile", type= str, help="input file name")
parser.add_argument("-o","--outfile", type= str, help="output file name", default="temp")
parser.add_argument("-A","--in_alphabet", type= str, help="output file name", default="asjp")
parser.add_argument("-c","--clust_algo", type= str, help="clustering algorithm name", default="labelprop")
parser.add_argument("-e","--eval", help="evaluate cognate clusters", action='store_true')
parser.add_argument("-s","--string_sim", type= str, help="string similarity algorithm name", default="ldn")
parser.add_argument("-N","--nexus", help="generate a nexus file if you want", action='store_true')
args = parser.parse_args()

infomap_threshold = args.thd

dataname = args.infile
outname = args.outfile

char_list = []

def read_data_ielex_type(fname):
    line_id = 0
    data_dict = defaultdict(lambda : defaultdict())
    cogid_dict = defaultdict(lambda : defaultdict())
    words_dict = defaultdict(lambda : defaultdict(list))
    langs_list = []
    concepts_list = []
    f = open(fname)
    header = f.readline().strip().split("\t")
    if args.eval:
        cogid_idx = header.index("cognate_class")
    word_idx = header.index(args.in_alphabet)
    for line in f:
        line = line.strip()
        arr = line.split("\t")

        lang, concept = arr[0], arr[2]
        
        if len(arr) < 5:
            print(line)
            continue
        if " " in arr[word_idx]:
            asjp_word = arr[word_idx].split(" ")
        else:
            asjp_word = arr[word_idx]

        if args.in_alphabet != "ipa":
            asjp_word = "".join(asjp_word).replace("0","").replace("_","").replace("+","")
            if args.in_alphabet == "asjp":
                asjp_word = cleanASJP(asjp_word)

        for ch in asjp_word:
            if ch not in char_list:
                char_list.append(ch)

        if len(asjp_word) < 1:
            continue

        data_dict[concept][line_id,lang] = asjp_word
        if args.eval:
            cogid = arr[cogid_idx]
            cogid = cogid.replace("-","")
            cogid = cogid.replace("?","")
            cogid_dict[concept][line_id,lang] = cogid
        
        words_dict[lang][concept].append(asjp_word)
        
        if lang not in langs_list:
            langs_list.append(lang)
        if concept not in concepts_list:
            concepts_list.append(concept)
        line_id += 1
    f.close()
    print(list(data_dict.keys()))
    return (data_dict, cogid_dict, words_dict, langs_list, concepts_list)

def igraph_clustering(matrix, threshold, method='labelprop'):
    """
    Method computes Infomap clusters from pairwise distance data.
    """
    random.seed(1234)
    G = igraph.Graph()
    vertex_weights = []
    for i in range(len(matrix)):
        G.add_vertex(i)
        vertex_weights += [0]
    
    # variable stores edge weights, if they are not there, the network is
    # already separated by the threshold
    weights = None
    for i,row in enumerate(matrix):
        for j,cell in enumerate(row):
            if i < j:
                if cell <= threshold:
                    G.add_edge(i, j, weight=1-cell, distance=cell)
                    weights = 'weight'

    if method == 'infomap':
        comps = G.community_infomap(edge_weights=weights,
                vertex_weights=None)
        
    elif method == 'labelprop':
        comps = G.community_label_propagation(weights=weights,
                initial=None, fixed=None)

    elif method == 'ebet':
        dg = G.community_edge_betweenness(weights=weights)
        oc = dg.optimal_count
        comps = False
        while oc <= len(G.vs):
            try:
                comps = dg.as_clustering(dg.optimal_count)
                break
            except:
                oc += 1
        if not comps:
            print('Failed...')
            comps = list(range(len(G.sv)))
            input()
    elif method == 'multilevel':
        comps = G.community_multilevel(return_levels=False)
    elif method == 'spinglass':
        comps = G.community_spinglass()

    D = {}
    for i,comp in enumerate(comps.subgraphs()):
        vertices = [v['name'] for v in comp.vs]
        for vertex in vertices:
            D[vertex] = i+1

    return D
    
def get_gj_pmi():
    lodict = defaultdict()
    f = open('pmi_model/sounds41.txt')
    sounds = np.array([x.strip() for x in f.readlines()])
    f.close()

    f = open('pmi_model/pmi-world.txt','r')
    l = f.readlines()
    f.close()
    logOdds = np.array([x.strip().split() for x in l],np.float)

    for i in range(len(sounds)):#Initiate sound dictionary
        for j in range(len(sounds)):
            lodict[sounds[i],sounds[j]] = logOdds[i,j]
    return lodict

def infomap_concept_evaluate_scores(d, lang_list):
    average_fscore = []
    f_scores = []
    bin_mat, n_clusters = [], 0
    if args.string_sim == "gj":
        lodict = get_gj_pmi()
        assert args.in_alphabet == "asjp"
    
    f_preds = open(args.outfile+"-"+args.string_sim+".cognates", "w")
    f_preds.write("gloss\tlanguage\t"+args.in_alphabet+"\tcognate class\n")
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
            score = 0.0
            if args.string_sim == "ldn":
                score = distances.ldn(w1, w2)
            elif args.string_sim == "prefix":
                score = distances.prefix(w1, w2)
            elif args.string_sim == "dice":
                score = distances.dice(w1, w2)
            elif args.string_sim == "nw":
                score = distances.nw(w1, w2)[0]
                score = 1.0 - (1.0/(1.0+np.exp(-score)))
            elif args.string_sim == "gj":
                score = distances.nw(w1, w2, lodict=lodict)[0]
                score = 1.0 - (1.0/(1.0+np.exp(-score)))

            ldn_dist_dict[l1][l2] = score
            ldn_dist_dict[l2][l1] = ldn_dist_dict[l1][l2]
        distMat = np.array([[ldn_dist_dict[ka][kb] for kb in langs] for ka in langs])
        clust = igraph_clustering(distMat, infomap_threshold, method=args.clust_algo)
        

        predicted_labels = defaultdict()
        predicted_labels_words = defaultdict()
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
            print(concept, len(langs), scores[0], scores[1], scores[2], len(set(clust.values())), len(set(truel)), sep="\t")
            f_scores.append(list(scores))
        n_clusters += len(set(clust.values()))
        if args.nexus:
            t = dict2binarynexus(predicted_labels, lang_list)
            bin_mat += t

    if args.eval:
        f_scores = np.mean(np.array(f_scores), axis=0)
        print(f_scores[0], f_scores[1], 2.0*f_scores[0]*f_scores[1]/(f_scores[0]+f_scores[1]))
    f_preds.close()
    return bin_mat


data_dict, cogid_dict, words_dict, lang_list, concepts_list = read_data_ielex_type(dataname)
print("Character list \n\n", char_list)
print("Length of character list ", len(char_list))

bin_mat= infomap_concept_evaluate_scores(data_dict, lang_list)

if args.nexus:
    nchar, nlangs = np.array(bin_mat).shape

    fw = open(outname+".nex","w")
    fw.write("begin data;"+"\n")
    fw.write("   dimensions ntax="+str(nlangs)+" nchar="+str(nchar)+";\nformat datatype=restriction interleave=no missing= ? gap=-;\nmatrix\n")

    for row, lang in zip(np.array(bin_mat).T, lang_list):
        #print(row,len(row), "\n")
        rowx = "".join([str(x) for x in row])
        fw.write(lang+"\t"+ rowx.replace("2","?")+"\n")
    fw.write(";\nend;")

