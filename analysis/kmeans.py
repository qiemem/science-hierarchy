import numpy as np

def initial_means(xs, num_clusters = 2):
    picks = set()
    for i in range(num_clusters):
        while len(picks) <= i:
            picks.add(np.random.randint(len(xs)))
    return np.array([xs[i] for i in picks])
    #return np.array(np.random.permutation(xs)[:num_clusters])

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
    clusters = get_clusters(xs, means, dist)
    print(map(len, clusters))
    return np.array(map(lambda xs : np.mean(xs, 0), clusters))

def kmeans(xs, k = 2, dist = lambda x,y : np.linalg.norm(x-y)):
    means = initial_means(xs, k)
    last_means = None
    while np.any(means != last_means):
        last_means = means
        means = update_means(xs, means)
        print([np.linalg.norm(x-y) for (x,y) in zip(last_means, means)])
    return means

def bisecting_kmeans(xs, k = 2, dist = lambda x,y : np.linalg.norm(x-y)):
    print(len(xs))
    if len(xs) < k:
        return Leaf(xs)
    else:
        bi = lambda ys : bisecting_kmeans(ys, k, dist)
        means = kmeans(xs, k, dist)
        clusters = get_clusters(xs, means, dist)
        return Tree(means, [bi(c) for c in clusters])

class Tree:
    def __init__(self, means, children):
        self.means = means
        self.children = children

    def dft(self, depth=-1):
        yield self
        if depth!=0:
            for c in self.children:
                for n in c.dft(depth-1):
                    yield n
    
    @property
    def depth(self):
        return 1 + max(c.depth for c in self.children)

    @property
    def num_nodes(self):
        return 1 + sum(c.num_nodes for c in self.children)

    @property
    def num_points(self):
        return sum(c.num_points for c in self.children)

class Leaf:
    def __init__(self, values):
        self.values = values
        self.depth = 1
        self.num_nodes = 1
        self.num_points = len(values)

    def dft(self, depth=-1):
        yield self
