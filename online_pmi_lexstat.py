from collections import defaultdict
import itertools as it
import sys, distances, igraph, utils
import numpy as np
import random, codecs
import DistanceMeasures as DM
from sklearn import metrics
from lingpy import *
random.seed(1234)

##TODO: Add a ML based estimation of distance or a JC model for distance between two sequences
##Separate clustering code.
##Add doculect distance as regularization

MAX_ITER = 10
tolerance = 0.001
infomap_threshold = 0.5
min_batch = 256
margin = 0.0

dataname = sys.argv[1]
#outname = sys.argv[2]
#fname = sys.argv[2]
char_list = []

def clean_word(w):
    w = w.replace("-","")
    w = w.replace(" ", "")
    w = w.replace("%","")
    w = w.replace("~","")
    w = w.replace("*","")
    w = w.replace("$","")
    w = w.replace("\"","")
    w = w.replace("K","k")
    w = w.replace("|","")
    w = w.replace(".","")
    w = w.replace("+","")
    w = w.replace("·","")
    w = w.replace("?","")
    w = w.replace("’","")
    w = w.replace("]","")
    w = w.replace("[","")
    w = w.replace("=","")
    w = w.replace("_","")
    w = w.replace("<","")
    w = w.replace(">","")
    #w = w.replace("‐","")
    #w = w.replace("ᶢ","")
   # w = w.replace("C","c")
   # w = w.replace("L","l")
   # w = w.replace("W","w")
   # w = w.replace("T","t")
    return w

def ipa2sca(w):
    return "".join(tokens2class(ipa2tokens(w), 'asjp')).replace("0","")

def read_data_ielex_type(fname):
    line_id = 0
    data_dict = defaultdict(lambda : defaultdict())
    cogid_dict = defaultdict(lambda : defaultdict())
    words_dict = defaultdict(lambda : defaultdict(list))
    langs_list = []
    concepts_list = []
    #f = codecs.open(fname, "r", "utf8")
    f = open(fname)
    header = f.readline().strip("\n").lower().split("\t")
    cogid_idx = header.index("cogid")
    if "doculect" in header:
        lang_idx = header.index("doculect")
    elif "language" in header:
        lang_idx = header.index("language")
    if "glottocode" in header:
        iso_idx = header.index("glottocode")
    else:
        iso_idx = header.index("iso_code")
    gloss_idx = header.index("concept")
    word_idx = header.index("asjp")
    for line in f:
        line = line.strip()
        arr = line.split("\t")
        lang = arr[lang_idx]
        #if lang in ["ELFDALIAN", "GUTNISH_LAU", "STAVANGERSK"]:
        #    continue
        concept = arr[gloss_idx]
        cogid = arr[cogid_idx]
        cogid = cogid.replace("-","")
        cogid = cogid.replace("?","")
        asjp_word = clean_word(arr[word_idx].split(",")[0])

        for ch in asjp_word:
            if ch not in char_list:
                char_list.append(ch)

        if len(asjp_word) < 1:
            continue

        data_dict[concept][line_id,lang] = asjp_word
        cogid_dict[concept][line_id,lang] = cogid
        words_dict[lang][concept].append(asjp_word)
        if lang not in langs_list:
            langs_list.append(lang)
        if concept not in concepts_list:
            concepts_list.append(concept)
        line_id += 1
    f.close()
    print(list(data_dict.keys()))
    print(concepts_list)
    return (data_dict, cogid_dict, words_dict, langs_list, concepts_list)

def optimize_gop_gep(pmi_dict, words_dict, langs_list, concepts_list):

    sum_lang_sim = 0.0
    gop_gep_dict = defaultdict(float)
    
    for gop in np.arange(-1.0, -3.2, -0.2):
        for scale_factor in np.arange(0.1,1.0,0.1):
            gep = gop*scale_factor
            sum_lang_sim = 0.0
            print(gop, gep)
            for l1, l2 in it.combinations(langs_list, r=2):
                num, denom = 0.0, 0.0
                for concept1, concept2 in it.product(concepts_list, concepts_list):
                    if concept1 not in words_dict[l1] or concept2 not in words_dict[l2]:
                        continue
                    else:
                        if concept1 == concept2:
                            for w1, w2 in it.product(words_dict[l1][concept1], words_dict[l2][concept2]):
                                num += distances.needleman_wunsch(w1, w2, scores=pmi_dict, gop=gop, gep=gep)[0]
                        elif concept1 != concept2:
                            for w1, w2 in it.product(words_dict[l1][concept1], words_dict[l2][concept2]):
                                denom += distances.needleman_wunsch(w1, w2, scores=pmi_dict, gop=gop, gep=gep)[0]
                sum_lang_sim += num/denom
                print(l1, l2, num/denom)
            gop_gep_dict[gop,gep] = sum_lang_sim
            print(gop, gep, sum_lang_sim)
    return max(gop_gep_dict, key=lambda key: gop_gep_dict[key])

def igraph_clustering(matrix, threshold, method='infomap'):
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
    
def infomap_concept_evaluate_scores(d, lodict, gop, gep):
    #fout = open("output.txt","w")
    average_fscore = []
    f_scores = []#defaultdict(list)
    bin_mat, n_clusters = None, 0
    for concept in d:
        ldn_dist_dict = defaultdict(lambda: defaultdict(float))
        langs = list(d[concept].keys())
        if len(langs) == 1:
            print(" Ignoring ", concept)
            continue
        scores, cognates = [], []
        #ex_langs = list(set(lang_list) - set(langs))
        for l1, l2 in it.combinations(langs, r=2):
            if d[concept][l1].startswith("-") or d[concept][l2].startswith("-"): continue
            w1, w2 = d[concept][l1], d[concept][l2]
            score = distances.needleman_wunsch(w1, w2, scores=lodict, gop=gop, gep=gep)[0]
            score = 1.0 - (1.0/(1.0+np.exp(-score)))
            ldn_dist_dict[l1][l2] = score
            ldn_dist_dict[l2][l1] = ldn_dist_dict[l1][l2]
        distMat = np.array([[ldn_dist_dict[ka][kb] for kb in langs] for ka in langs])
        clust = igraph_clustering(distMat, infomap_threshold, method='labelprop')
        
        
        predicted_labels = defaultdict()
        predicted_labels_words = defaultdict()
        for k, v in clust.items():
            predicted_labels[langs[k]] = v
            predicted_labels_words[langs[k],d[concept][langs[k]]] = v
        
        #print(concept,"\n",predicted_labels_words)
        predl, truel = [], []
        for l in langs:
            truel.append(cogid_dict[concept][l])
            predl.append(predicted_labels[l])
        scores = DM.b_cubed(truel, predl)
        
        #scores = metrics.f1_score(truel, predl, average="micro")
        print(concept, len(langs), *scores, len(set(clust.values())), len(set(truel)), sep="\t")
        f_scores.append(list(scores))
        n_clusters += len(set(clust.values()))
        #t = utils.dict2binarynexus(predicted_labels, ex_langs, lang_list)
        #print(concept, "\n",t)
        #print("No. of clusters ", n_clusters)
    #print(np.mean(np.array(f_scores), axis=0))
    f_scores = np.mean(np.array(f_scores), axis=0)
    print(f_scores[0], f_scores[1], 2.0*f_scores[0]*f_scores[1]/(f_scores[0]+f_scores[1]))
    return bin_mat

def calc_pmi(alignment_dict, char_list, scores, initialize=True):
    sound_dict = defaultdict(float)
    relative_align_freq = 0.0
    relative_sound_freq = 0.0
    count_dict = defaultdict(float)
    
    if initialize == True:
        for c1, c2 in it.product(char_list, repeat=2):
            if c1 == "-" or c2 == "-":
                continue
            count_dict[c1,c2] += 0.001
            count_dict[c2,c1] += 0.001
            sound_dict[c1] += 0.001
            sound_dict[c2] += 0.001
            relative_align_freq += 0.001
            relative_sound_freq += 0.002

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
    
    for a in count_dict.keys():
        m = count_dict[a]
        if m <=0: print(a, m)
        assert m>0

        num = np.log(m)-np.log(relative_align_freq)
        denom = np.log(sound_dict[a[0]])+np.log(sound_dict[a[1]])-(2.0*np.log(relative_sound_freq))
        val = num - denom
        count_dict[a] = val
    
    return count_dict

def lexstat_concept_evaluate_scores(d, lodict, gop, gep):
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
        #ex_langs = list(set(lang_list) - set(langs))
        for l1, l2 in it.combinations(langs, r=2):
            if d[concept][l1].startswith("-") or d[concept][l2].startswith("-"): continue
            w1, w2 = d[concept][l1], d[concept][l2]
            x, lang1 = l1
            y, lang2 = l2
            score = distances.needleman_wunsch(w1, w2, scores=lodict[lang1,lang2], gop=gop, gep=gep)[0]
            score = 1.0 - (1.0/(1.0+np.exp(-score)))
            ldn_dist_dict[l1][l2] = score
            ldn_dist_dict[l2][l1] = ldn_dist_dict[l1][l2]
        distMat = np.array([[ldn_dist_dict[ka][kb] for kb in langs] for ka in langs])
        clust = igraph_clustering(distMat, infomap_threshold, method='labelprop')
        
        
        predicted_labels = defaultdict()
        predicted_labels_words = defaultdict()
        for k, v in clust.items():
            predicted_labels[langs[k]] = v
            predicted_labels_words[langs[k],d[concept][langs[k]]] = v
        
        #print(concept,"\n",predicted_labels_words)
        predl, truel = [], []
        for l in langs:
            truel.append(cogid_dict[concept][l])
            predl.append(predicted_labels[l])
        scores = DM.b_cubed(truel, predl)
        
        #scores = metrics.f1_score(truel, predl, average="micro")
        print(concept, len(langs), *scores, len(set(clust.values())), len(set(truel)),sep="\t")
        f_scores.append(list(scores))
        n_clusters += len(set(clust.values()))
        #t = utils.dict2binarynexus(predicted_labels, ex_langs, lang_list)
        #print(concept, "\n",t)
        #print("No. of clusters ", n_clusters)
    #print(np.mean(np.array(f_scores), axis=0))
    f_scores = np.mean(np.array(f_scores), axis=0)
    print(f_scores[0], f_scores[1], 2.0*f_scores[0]*f_scores[1]/(f_scores[0]+f_scores[1]))
    return bin_mat


data_dict, cogid_dict, words_dict, langs_list, concepts_list = read_data_ielex_type(dataname)
print("Character list \n\n", char_list)
print("Length of character list ", len(char_list))
print("Language list ", langs_list)
word_list = []

for concept in data_dict:
    print(concept)
    words = []
    for idx in data_dict[concept]:
        words.append(data_dict[concept][idx])
    for x, y in it.combinations(words, r=2):
        word_list += [[x,y]]
        #if distances.needleman_wunsch(x, y, scores={}, gop=-1,gep=-0.5)[0] > 0.0:
        #if distances.ldn(x, y) <=0.5:
        #    word_list += [[x,y]]

#word_list = [line.strip().split()[0:2] for line in open(fname).readlines()]
#char_list = [line.strip() for line in open("sounds41.txt").readlines()]


pmidict = None
n_examples, n_updates, alpha = len(word_list), 0, 0.75
n_wl = len(word_list)
print("Size of initial list ", n_wl)

pmidict = defaultdict(float)

GOP, GEP = -2.5, -1.75

for n_iter in range(MAX_ITER):
    random.shuffle(word_list)
    pruned_wl = []
    n_zero = 0.0
    print("Iteration ", n_iter)
    for idx in range(0, n_wl, min_batch):
        wl = word_list[idx:idx+min_batch]
        eta = np.power(n_updates+2, -alpha)
        algn_list, scores = [], []
        for w1, w2 in wl:
            #print(w1,w2,sc)
            if not pmidict:
                s, alg = distances.needleman_wunsch(w1, w2, scores={}, gop=GOP, gep=GEP)
            else:
                s, alg = distances.needleman_wunsch(w1, w2, scores=pmidict, gop=GOP, gep=GEP)
            if s <= margin:
                n_zero += 1.0
                continue
            #s = s/max(len(w1), len(w2))
            algn_list.append(alg)
            scores.append(distances.sigmoid(s))
            #pruned_wl.append([w1[::-1], w2[::-1], s])	  
            pruned_wl.append([w1, w2])	  
        mb_pmi_dict = calc_pmi(algn_list, char_list, scores, initialize=True)
        for k, v in mb_pmi_dict.items():
            pmidict_val = pmidict[k]
            pmidict[k] = (eta*v) + ((1.0-eta)*pmidict_val)
        n_updates += 1
    print("Non zero examples ", n_wl, n_wl-n_zero, " number of updates ", n_updates)
    word_list = pruned_wl[:]
    n_wl = len(word_list)
    #infomap_concept_evaluate_scores(data_dict, pmidict, -2.5, -1.75)

#print(pmidict)

#GOP, GEP = optimize_gop_gep(pmidict, words_dict, langs_list, concepts_list)

lexstat_scores = defaultdict(lambda: defaultdict(float))
denom_scores = defaultdict(lambda: defaultdict(float))

cache_scores = defaultdict()

for l1, l2 in it.combinations_with_replacement(langs_list, r=2):# can optimize by operating on a set of words
    print(l1, l2)
    for concept in concepts_list:
        if concept not in words_dict[l1] or concept not in words_dict[l2]:
            continue
        else:
            for w1, w2 in it.product(words_dict[l1][concept], words_dict[l2][concept]):
                if (w1,w2) in cache_scores:
                    algn = cache_scores[w1,w2]
                else:
                    algn = distances.needleman_wunsch(w1, w2, scores={}, gop=GOP, gep=GEP)[1]
                    cache_scores[w1,w2] = algn                            
                    cache_scores[w2,w1] = algn
                for x, y in algn:
                    if x == "-" or y == "-":
                        continue
                    else:
                        lexstat_scores[l1,l2][x,y] += 1.0
                        lexstat_scores[l1,l2][y,x] += 1.0


for l1, l2 in it.combinations_with_replacement(langs_list, r=2):
    shuffle_list = concepts_list[:]
    print(l1, l2)
    for i in range(100):
        #print("Iteration ",i)
        random.shuffle(shuffle_list)
        for c1, c2 in zip(shuffle_list, concepts_list):
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

for l1, l2 in it.combinations_with_replacement(langs_list, r=2):
    for x, y in it.product(char_list, char_list):
        lang_score = 0.0
        if (x,y) in lexstat_scores[l1,l2] and (x,y) in denom_scores:
            lang_score = np.log(lexstat_scores[l1,l2][x,y])-np.log(denom_scores[l1,l2][x,y])
        lexstat_scores[l1,l2][x,y] = lang_score+ (0.5*pmidict[x,y])
        lexstat_scores[l2,l1][x,y] = lexstat_scores[l1,l2][x,y]

print("\nPMI scores\n")
infomap_concept_evaluate_scores(data_dict, pmidict, GOP, GEP)

print("\nLexStat scores\n")
lexstat_concept_evaluate_scores(data_dict, lexstat_scores, GOP, GEP)

sys.exit(1)

nchar, nlangs = np.array(bin_mat).shape

fw = open(outname+".nex","w")
fw.write("begin data;"+"\n")
fw.write("   dimensions ntax="+str(nlangs)+" nchar="+str(nchar)+";\nformat datatype=restriction interleave=no missing= ? gap=-;\nmatrix\n")

for row, lang in zip(np.array(bin_mat).T, lang_list):
    #print(row,len(row), "\n")
    rowx = "".join([str(x) for x in row])
    fw.write(lang, "\t", rowx.replace("2","?")+"\n")
print(";\nend;")

    
    
    
    
