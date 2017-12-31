import random, igraph

def upgma(distmat, threshold, names):
    """
    UPGMA
    :param distmat: distance matrix
    :type distmat: list or numpy.core.ndarray
    :param threshold: threshold for cutting the treee
    :type threshold: float
    :param names: name of the taxa
    :type names: list
    :return: clusters
    :rtype: dict
    """

    # create cluster for individual nodes
    clusters = collections.defaultdict(list)
    for i in range(len(distmat)):
        clusters[i] = [i]

    # call internal upgma
    clust = upgma_int(clusters, distmat, threshold)
    
    # assign names to the clusters
    for key in clust:
        clust[key] = [names[i] for i in clust[key]]
    return clust


def upgma_int(
        clusters,
        matrix,
        threshold
        ):
    """
    Internal upgma implementation
    :param clusters: dictionary of clusters
    :type clusters: dict
    :param matrix: distance matrix
    :type matrix: list or numpy.core.ndarry
    :param threshold: threshold for cutting the upgma tree
    :type threshold: float
    :return: clusters
    :rtype: dict
    """

    done = False

    while done is False:

        # check if first termination condition is reached
        if len(clusters) == 1:
            done = True

        else:
            # dictionary with indices of scores
            sc_ind = collections.defaultdict(float)
            # calculate score of existing clusters
            for (i, valA), (j, valB) in itertools.permutations(clusters.items(), 2):
                s = 0.0
                ct = 0
                for vA, vB in itertools.product(valA, valB):
                    s += matrix[vA][vB]
                    ct += 1
                sc_ind[(i, j)] = (s / ct)

            minimum_ind = min(sc_ind, key=sc_ind.get)

            # check if second termination condition is reached
            # everything left above threshold
            if sc_ind[minimum_ind] <= threshold:
                # form new cluster
                idx, jdx = minimum_ind
                clusters[idx] += clusters[jdx]
                del clusters[jdx]
            else:
                done = True

    return clusters


def single_linkage(distmat, threshold, names):
    """
    single linkage clustering
    :param distmat: distance matrix
    :type distmat: list or numpy.core.ndarray
    :param threshold: threshold for cutting the treee
    :type threshold: float
    :param names: name of the taxa
    :type names: list
    :return: clusters
    :rtype: dict
    """

    # create cluster for individual nodes
    clusters = collections.defaultdict(list)
    for i in range(len(distmat)):
        clusters[i] = [i]

    # call internal upgma
    clust = single_linkage_int(clusters, distmat, threshold)

    # assign names to the clusters
    for key in clust:
        clust[key] = [names[i] for i in clust[key]]
    return clust


def single_linkage_int(clusters, matrix, threshold):
    """
    internal implementation of single linkage clustering
    :param clusters: dictionary of clusters
    :type clusters: dict
    :param matrix: distance matrix
    :type matrix: list or numpy.core.ndarry
    :param threshold: threshold for cutting the upgma tree
    :type threshold: float
    :return: clusters
    :rtype: dict
    """
    done = False

    while done is False:

        # check if first termination condition is reached
        if len(clusters) == 1:
            done = True

        else:
            # dictionary with indices of scores
            sc_ind = collections.defaultdict(float)
            # calculate score of existing clusters
            for (i, valA), (j, valB) in itertools.permutations(clusters.items(), 2):
                sc_ind[(i, j)] = float("inf")
                for vA, vB in itertools.product(valA, valB):
                    if matrix[vA][vB] < sc_ind[(i, j)]:
                        sc_ind[(i, j)] = matrix[vA][vB]

            minimum_ind = min(sc_ind, key=sc_ind.get)

            # check if second termination condition is reached
            # everything left above threshold
            if sc_ind[minimum_ind] <= threshold:
                # form new cluster
                idx, jdx = minimum_ind
                clusters[idx] += clusters[jdx]
                del clusters[jdx]
            else:
                done = True

    return clusters

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
