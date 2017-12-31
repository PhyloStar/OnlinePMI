from collections import defaultdict
import numpy as np
import itertoosl as it

def optimize_gop_gep(pmi_dict, words_dict, langs_list, concepts_list):
    sum_lang_sim = 0.0
    gop_gep_dict = defaultdict(float)
    
    for gop in np.arange(-2.0, -3.0, -0.2):
        for scale_factor in np.arange(0.5,1.0,0.1):
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
                                num += distances.sigmoid(distances.needleman_wunsch(w1, w2, scores=pmi_dict, gop=gop, gep=gep)[0])
                        elif concept1 != concept2:
                            for w1, w2 in it.product(words_dict[l1][concept1], words_dict[l2][concept2]):
                                denom += distances.sigmoid(distances.needleman_wunsch(w1, w2, scores=pmi_dict, gop=gop, gep=gep)[0])
                sum_lang_sim += num/denom
                print(l1, l2, num/denom)
            gop_gep_dict[gop,gep] = sum_lang_sim
            print(gop, gep, sum_lang_sim)
    return max(gop_gep_dict, key=lambda key: gop_gep_dict[key])
    
def scipy_optimize_gop_gep(pmi_dict, words_dict, langs_list, concepts_list, gop, gep):
    from scipy.optimize import minimize

    def lang_dist_func(GP):
        sum_lang_sim = 0.0

        for l1, l2 in it.combinations(langs_list, r=2):
            num, denom = 0.0, 0.0
            #print(l1, l2)
            for concept1, concept2 in it.product(concepts_list, concepts_list):
                if concept1 not in words_dict[l1] or concept2 not in words_dict[l2]:
                    continue
                else:
                    if concept1 == concept2:
                        for w1, w2 in it.product(words_dict[l1][concept1], words_dict[l2][concept2]):
                            #num += distances.sigmoid(distances.needleman_wunsch(w1, w2, scores=pmi_dict, gop=GP[0], gep=GP[1])[0])
                            num += max(distances.needleman_wunsch(w1, w2, scores=pmi_dict, gop=GP[0], gep=GP[1])[0],0)
                            #num += distances.needleman_wunsch(w1, w2, scores=pmi_dict, gop=GP[0], gep=GP[1])[0]
                    elif concept1 != concept2:
                        for w1, w2 in it.product(words_dict[l1][concept1], words_dict[l2][concept2]):
                            #denom += distances.sigmoid(distances.needleman_wunsch(w1, w2, scores=pmi_dict, gop=GP[0], gep=GP[1])[0])
                            denom += max(distances.needleman_wunsch(w1, w2, scores=pmi_dict, gop=GP[0], gep=GP[1])[0],0)
                            #denom += distances.needleman_wunsch(w1, w2, scores=pmi_dict, gop=GP[0], gep=GP[1])[0]
            sum_lang_sim += denom - num
        return sum_lang_sim
    GP = np.array([gop, gep])
    res = minimize(lang_dist_func, GP, method='powell',tol=1e-1, options={'disp': True})

    return res.x
