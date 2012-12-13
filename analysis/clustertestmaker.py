import random
import numpy as np

def make_test(tree, vecs_to_docs, n_questions = 50,
        dist = lambda x,y : np.linalg.norm(x-y)):
    trees = [t for t in tree.dft() if len(t.children)>0 and len(t)<2000]
    vecs = vecs_to_docs.keys()
    #docs_to_vecs = {d:v for (v,ds) in vecs_to_docs.items() for d in ds}
    questions = set()
    while len(questions) < n_questions:
        cluster = random.choice(trees)
        in_cluster = [v for v in vecs if v in cluster]
        if len(in_cluster)<10:
            continue
        x, a = random.sample(in_cluster, 2)
        xvec = np.array(x)
        d = dist(xvec,a)
        in_cluster_set = set(in_cluster)
        out_cluster = [v for v in vecs if v not in in_cluster_set and dist(xvec,v)==d]
        if len(out_cluster) == 0:
            continue
        b = random.choice(out_cluster)
        xdoc = random.choice(vecs_to_docs[x])
        adoc = random.choice(vecs_to_docs[a])
        bdoc = random.choice(vecs_to_docs[b])
        questions.add((xdoc, adoc, bdoc, d, len(in_cluster)))
    return list(questions)
    
