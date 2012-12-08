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
    print([len(c) for c in clusters])
    if any(len(c)==0 for c in clusters):
        raise Exception
    return np.array(map(lambda x : np.mean(x,0), clusters))

#def typed_mean(xs):
    #return np.mean(xs,0)>=.5
    #m = np.mean(xs,0)
    #try:
    #    if xs[0].dtype.name == 'bool':
    #        return m>=.5
    #except IndexError:
    #    pass
    #return m

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
        return Tree(np.mean(xs, 0), [])
    else:
        bi = lambda ys : bisecting_kmeans(ys, k, dist)
        means = kmeans(xs, k, dist)
        clusters = get_clusters(xs, means, dist)
        return Tree(np.mean(xs,0), [bi(c) for c in clusters])

class Tree:
    def __init__(self, mean, children, parent=None):
        self.parent = parent
        self.mean = mean
        self.children = children

    def __contains__(self, point):
        if self.parent:
            return self.closest_child(point) == self and point in self.parent
        else:
            return True

    def closest_child(self, point):
        i = np.argmin([np.linalg.norm(point-c.mean) for c in self.children])
        return self.children[i]

    def dft(self, depth=-1):
        yield self
        if depth!=0:
            for c in self.children:
                for n in c.dft(depth-1):
                    yield n
    
    @property
    def depth(self):
        if len(self.children):
            return 1 + max(c.depth for c in self.children)
        else:
            return 1

    def __len__(self):
        return 1 + sum(c.num_nodes for c in self.children)
