from collections import defaultdict
import itertools
import numpy as np
from lingpy import *

def cleanASJP(w):
    w = w.replace("%","")
    w = w.replace("~","")
    w = w.replace("*","")
    w = w.replace("\"","")
    w = w.replace("$","")
    return w

def writeNexus(dm,f):
    l=len(dm)
    f.write("#nexus\n"+
                  "\n")
    f.write("BEGIN Taxa;\n")
    f.write("DIMENSIONS ntax="+str(l)+";\n"
                "TAXLABELS\n"+
                "\n")
    i=0
    for ka in dm:
        f.write("["+str(i+1)+"] '"+ka+"'\n")
        i=i+1
    f.write(";\n"+
               "\n"+
               "END; [Taxa]\n"+
               "\n")
    f.write("BEGIN Distances;\n"
            "DIMENSIONS ntax="+str(l)+";\n"+
            "FORMAT labels=left;\n")
    f.write("MATRIX\n")
    i=0
    for ka in dm:
        row="["+str(i+1)+"]\t'"+ka+"'\t"
        for kb in dm:
            row=row+str(dm[ka][kb])+"\t"
        f.write(row+"\n")
    f.write(";\n"+
    "END; [Distances]\n"+
    "\n"+
    "BEGIN st_Assumptions;\n"+
    "    disttransform=NJ;\n"+
    "    splitstransform=EqualAngle;\n"+
    "    SplitsPostProcess filter=dimension value=4;\n"+
    "    autolayoutnodelabels;\n"+
        "END; [st_Assumptions]\n")
    f.flush()

def dict2binarynexus1(d, ex_langs, langs):
    binArr = []
    x = defaultdict(lambda: defaultdict(int))
    for lang, clusterID in d.items():
        x[clusterID][lang] = 1

    for clusterID in x.keys():
        temp = []
        for lang in langs:
            if lang in ex_langs: temp.append(2)
            else:
                temp += [x[clusterID][lang]]
        binArr.append(temp)
    return binArr
    
def dict2binarynexus(d, lang_list):
    binArr = []
    x = defaultdict(lambda: defaultdict(int))
    LANGS = []
    for langID, clusterID in d.items():
        if langID[1] not in LANGS: LANGS.append(langID[1])
        x[clusterID][langID[1]] = 1

    for clusterID in x.keys():
        temp = []
        for lang in lang_list:
            if lang not in LANGS: temp.append("2")
            else:
                temp += [x[clusterID][lang]]
        binArr.append(temp)
    return binArr

def ipa2sca(w):
    return "".join(tokens2class(ipa2tokens(w), 'asjp')).replace("0","").replace("_","")

def read_data_ielex_type(fname, reverse=False, in_alphabet="asjp"):
    char_list = []
    line_id = 0
    data_dict = defaultdict(lambda : defaultdict())
    cogid_dict = defaultdict(lambda : defaultdict())
    words_dict = defaultdict(lambda : defaultdict(list))
    langs_list, concepts_list = [], []

    f = open(fname)
    header = f.readline().strip("\n").lower().split("\t")
    
    cogid_idx = header.index("cogid")
    word_idx = header.index(in_alphabet)
    if "doculect" in header:
        lang_idx = header.index("doculect")
    elif "language" in header:
        lang_idx = header.index("language")
    if "glottocode" in header:
        iso_idx = header.index("glottocode")
    else:
        iso_idx = header.index("iso_code")
    gloss_idx = header.index("concept")
    print("Reading asjp alphabet in ", word_idx)
    for line in f:
        line = line.strip()
        arr = line.split("\t")
        lang, iso, concept = arr[lang_idx], arr[iso_idx], arr[gloss_idx]
        
        if len(arr) < 4:
            continue

        if " " in arr[word_idx]:
            asjp_word = arr[word_idx].split(" ")
        else:
            asjp_word = arr[word_idx]

        if in_alphabet != "ipa":
            asjp_word = "".join(asjp_word).replace("0","").replace("_","").replace("+","")
            if in_alphabet == "asjp":
                asjp_word = cleanASJP(asjp_word)

        for ch in asjp_word:
            if ch not in char_list:
                char_list.append(ch)

        if len(asjp_word) < 1:
            continue
        if reverse:
            data_dict[concept][line_id,lang] = asjp_word[::-1]
        else:
            data_dict[concept][line_id,lang] = asjp_word

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

    return (data_dict, cogid_dict, words_dict, langs_list, concepts_list, char_list)

def read_pmidict(pmi_fname):
    scores = {}
    for line in open(pmi_fname, "r"):
        x, y, s = line.replace("\n","").split("\t")
        scores[x,y]=float(s)
    return scores
