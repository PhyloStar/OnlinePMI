from lingpy import *
from lingpy.sequence.sound_classes import asjp2tokens

import glob, sys

def ipa2sca(w, sound_class="asjp"):
    return " ".join(tokens2class(ipa2tokens(w), sound_class))

def asjp2sca(w, sound_class="asjp"):
    return " ".join(tokens2class(asjp2tokens(w), sound_class))

if sys.argv[1] == "asjp":
    rc(schema="asjp")

for fname in glob.iglob("/home/taraka/cognate_ensemble/ipa_data/sino*tsv"):
    f = open(fname, "r")
    print("Processing ", fname)
    fout = open(fname+".uniform", "w")
    header = f.readline().split("\t")[:3]
    if sys.argv[1] != "asjp":
        header+= ["ipa", "asjp", "sca", "dolgo", "cognate_class"]
    else:
        header+= ["asjp", "sca", "dolgo", "cognate_class"]
    print("\t".join(header), file=fout)
    for line in f:
        arr = line.strip().split("\t")
        out_line = [arr[0], arr[1], arr[2]]
        word = arr[5]
        cognate_class = arr[6]
        if sys.argv[1] != "asjp":
            out_line += [" ".join(ipa2tokens(word)), ipa2sca(word, sound_class="asjp"), ipa2sca(word, sound_class="sca"), ipa2sca(word, sound_class="dolgo"), cognate_class]
        #print(out_line, word, asjp2tokens(word))
        else:
            out_line += [" ".join(asjp2tokens(word)), asjp2sca(word, sound_class="sca"), asjp2sca(word, sound_class="dolgo"), cognate_class]
        print(line)
        print(out_line)
        print("\t".join(out_line), file=fout)
    f.close()
    fout.close()
