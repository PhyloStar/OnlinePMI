import collections
import itertools
import numpy as np

#Add dice, jaccard, prefix

def sigmoid(score):
    return 1.0/(1.0+np.exp(-score))

def ldn(a, b):
    """
    Leventsthein distance normalized
    :param a: word
    :type a: str
    :param b: word
    :type b: str
    :return: distance score
    :rtype: float
    """
    m = [];
    la = len(a) + 1;
    lb = len(b) + 1
    for i in range(0, la):
        m.append([])
        for j in range(0, lb): m[i].append(0)
        m[i][0] = i
    for i in range(0, lb): m[0][i] = i
    for i in range(1, la):
        for j in range(1, lb):
            s = m[i - 1][j - 1]
            if (a[i - 1] != b[j - 1]): s = s + 1
            m[i][j] = min(m[i][j - 1] + 1, m[i - 1][j] + 1, s)
    la = la - 1;
    lb = lb - 1
    return float(m[la][lb])/ float(max(la, lb))
    
def LD(x,y,scores=None):
    """
    x is target, y is source
    Needleman-Wunsch algorithm for pairwise string alignment
    with affine gap penalties.
    'scores' must be a dictionary with all symbol pairs as keys
    and match scores as values.
    gop and gep are gap penalties for opening/extending a gap.
    Returns the alignment score and one optimal alignment.
    """
    n,m = len(x),len(y)
    dp = np.zeros((n+1,m+1))
    pointers = np.zeros((n+1,m+1),np.int32)
    for i in range(1,n+1):
        if not scores:
            dp[i,0] = dp[i-1,0]+1
        else:
            dp[i,0] = dp[i-1,0]+scores['-',x[i-1]]
        pointers[i,0]=1
    for j in range(1,m+1):
        if not scores:
            dp[0,j] = dp[0,j-1]+1
        else:
            dp[0,j] = dp[0,j-1]+scores[y[j-1],'-']
        pointers[0,j]=2
    for i in range(1,n+1):
        for j in range(1,m+1):
            if not scores:
                match = dp[i-1,j-1]
                if x[i-1] != y[j-1]:
                    match = match+1                
                insert = dp[i-1,j]+1
                delet = dp[i,j-1]+1
            else:
                match = dp[i-1,j-1]+scores[x[i-1],y[j-1]]
                insert = dp[i-1,j]+scores['-',x[i-1]]
                delet = dp[i,j-1]+scores[y[j-1],'-']
            min_score = min([match,insert,delet])
            dp[i,j] = min_score
            pointers[i,j] = [match,insert,delet].index(min_score)
    alg = []
    i,j = n,m
    while(i>0 or j>0):
        pt = pointers[i,j]
        if pt==0:
            i-=1
            j-=1
            alg = [[x[i],y[j]]]+alg
        if pt==1:
            i-=1
            alg = [[x[i],'-']]+alg
        if pt==2:
            j-=1
            alg = [['-',y[j]]]+alg
    return dp[-1,-1], alg


def nw(x,y,scores=None,gop=-2.5,gep=-1.75):
    """
    Needleman-Wunsch algorithm for pairwise string alignment
    with affine gap penalties.
    'scores' must be a dictionary with all symbol pairs as keys
    and match scores as values.
    gop and gep are gap penalties for opening/extending a gap.
    Returns the alignment score and one optimal alignment.
    """
    n,m = len(x),len(y)
    dp = np.zeros((n+1,m+1))
    pointers = np.zeros((n+1,m+1),np.int32)
    for i in range(1,n+1):
        dp[i,0] = dp[i-1,0]+(gep if i>1 else gop)
        pointers[i,0]=1
    for j in range(1,m+1):
        dp[0,j] = dp[0,j-1]+(gep if j>1 else gop)
        pointers[0,j]=2
    for i in range(1,n+1):
        for j in range(1,m+1):
            if not scores:
                if x[i-1] == y[j-1]:
                    match = dp[i-1,j-1]+1
                else:
                    match = dp[i-1,j-1]-1
            else:
                match = dp[i-1,j-1]+scores[x[i-1],y[j-1]]
            insert = dp[i-1,j]+(gep if pointers[i-1,j]==1 else gop)
            delet = dp[i,j-1]+(gep if pointers[i,j-1]==2 else gop)
            max_score = max([match,insert,delet])
            dp[i,j] = max_score
            pointers[i,j] = [match,insert,delet].index(max_score)
    alg = []
    i,j = n,m
    #print(pointers)
    while(i>0 or j>0):
        pt = pointers[i,j]
        if pt==0:
            i-=1
            j-=1
            alg = [[x[i],y[j]]]+alg
        if pt==1:
            i-=1
            alg = [[x[i],'-']]+alg
        if pt==2:
            j-=1
            alg = [['-',y[j]]]+alg
    
    #print(alg)
    #print(dp)
    return dp[-1,-1], alg

def nonp_nw(x,y,scores=None,gop=-2.5,gep=-1.75):
    """
    Needleman-Wunsch algorithm for pairwise string alignment
    with affine gap penalties.
    'scores' must be a dictionary with all symbol pairs as keys
    and match scores as values.
    gop and gep are gap penalties for opening/extending a gap.
    Returns the alignment score and one optimal alignment.
    """
    n,m = len(x),len(y)
    #dp = np.zeros((n+1,m+1))
    dp = [[0]*(m+1) for _ in range(n+1)]
    #dp = [[0.0]*(m+1)]*(n+1)
    #pointers = np.zeros((n+1,m+1),np.int32)
    #pointers = [[0]*(m+1)]*(n+1)
    pointers = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1,n+1):
        dp[i][0] = dp[i-1][0]+(gep if i>1 else gop)
        pointers[i][0]=1
    for j in range(1,m+1):
        dp[0][j] = dp[0][j-1]+(gep if j>1 else gop)
        pointers[0][j]=2
    for i in range(1,n+1):
        for j in range(1,m+1):
            if not scores:
                if x[i-1] == y[j-1]:
                    match = dp[i-1][j-1]+1
                else:
                    match = dp[i-1][j-1]-1
            else:
                match = dp[i-1][j-1]+scores[x[i-1]][y[j-1]]
            insert = dp[i-1][j]+(gep if pointers[i-1][j]==1 else gop)
            delet = dp[i][j-1]+(gep if pointers[i][j-1]==2 else gop)
            max_score = max([match,insert,delet])
            dp[i][j] = max_score
            pointers[i][j] = [match,insert,delet].index(max_score)
    alg = []
    i,j = n,m
    print(x, y)
    print(dp)
    print(pointers)
    while(i>0 or j>0):
        pt = pointers[i][j]
        if pt==0:
            i-=1
            j-=1
            alg = [[x[i],y[j]]]+alg
        if pt==1:
            i-=1
            alg = [[x[i],'-']]+alg
        if pt==2:
            j-=1
            alg = [['-',y[j]]]+alg
    print(alg)
    return dp[-1][-1], alg


def prefix(a,b):
    pref = 0.0
    for x, y in zip(a, b):
        if x == y:
            pref += 1.0
        else:
            break
    return 1.0 - (pref/max(len(a), len(b)))

def dice(a,b):
    bia = set(list(zip(a[:-1],a[1:])))
    bib = set(list(zip(b[:-1],b[1:])))
    if len(a) == 1 or len(b) == 1:
        return 1.0
    return 1.0-(2.0*len(bia&bib)/(len(bia)+len(bib)))

def needleman_wunsch(seq_a, seq_b, scores={}, gop=-2.5, gep=-1.75):
    """
    Align two sequences using a flavour of the Needleman-Wunsch algorithm with
    fixed gap opening and gap extension penalties, attributed to Gotoh (1994).

    The scores arg should be a (char_a, char_b): score dict; if a char pair is
    missing, 1/-1 are used as match/mismatch scores.

    Return the best alignment score and one optimal alignment.
    """
    matrix = {}  # (x, y): (score, back)

    for y in range(len(seq_b) + 1):
        for x in range(len(seq_a) + 1):
            cands = []  # [(score, back), ..]

            if x > 0:
                score = matrix[(x-1, y)][0] \
                    + (gep if matrix[(x-1, y)][1] == '←' else gop)
                cands.append((score, '←'))

            if y > 0:
                score = matrix[(x, y-1)][0] \
                    + (gep if matrix[(x, y-1)][1] == '↑' else gop)
                cands.append((score, '↑'))

            if x > 0 and y > 0:
                if (seq_a[x-1], seq_b[y-1]) in scores:
                    score = scores[(seq_a[x-1], seq_b[y-1])]
                else:
                    score = 1 if seq_a[x-1] == seq_b[y-1] else -1
                score += matrix[(x-1, y-1)][0]
                cands.append((score, '.'))
            elif x == 0 and y == 0:
                cands.append((0.0, '.'))

            matrix[(x, y)] = max(cands)

    alignment = []

    while (x, y) != (0, 0):
        if matrix[(x, y)][1] == '←':
            alignment.append((seq_a[x-1], '-'))
            x -= 1
        elif matrix[(x, y)][1] == '↑':
            alignment.append(('-', seq_b[y-1]))
            y -= 1
        else:
            alignment.append((seq_a[x-1], seq_b[y-1]))
            x, y = x-1, y-1

    return matrix[(len(seq_a), len(seq_b))][0], tuple(reversed(alignment))

