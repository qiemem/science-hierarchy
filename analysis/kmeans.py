import numpy as np

def initial_means(xs):
    return np.random.permutation(xs)[:2]

def get_clusters(xs, means, dist = lambda x,y : np.linalg.norm(x-y)):
    """
    Input:
    xs - The data as an array-like object, e.g. a list of vectors, a matrix
        with each row being a data point.
    means - The current means as a list of vectors.
    dist - The distance measure to use. Euclidean distance by default.
    """
    clusters = [[] for _ in means]
    for x in xs:
        clusters[np.argmin([dist(x,m) for m in means])].append(x)
    return map(np.array, clusters)

def update_means(xs, means, dist = lambda x,y : np.linalg.norm(x-y)):
    """
    Input:
    xs - The data as an array-like object, e.g. a list of vectors, a matrix
        with each row being a data point.
    means - The current means as a list of vectors.
    dist - The distance measure to use. Euclidean distance by default.

    Output:
    The means of the clusters implied by the input means.
    """
    return np.array(map(lambda xs : np.mean(xs, 0), get_clusters(xs, means, dist)))
