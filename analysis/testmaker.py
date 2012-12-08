import random
import numpy as np

def make_dist_test(docs, vecs_to_docs, n_questions = 50):
    """
    Given docs (list) and vecs_to_docs (dict) from arxiv.py, output a list 
    of n_questions many distance-test questions.

    Question format: tuple of (x,test,control,test-dist,control-dist)
    """

    vecs = vecs_to_docs.keys()
    docs_to_vecs = {}
    for vec, ds in vecs_to_docs.items():
        for doc in ds:
            docs_to_vecs[doc] = vec
    k = len(docs)
    question_list = []
    i = 0
    while i < n_questions:

        [x,a,b]=random.sample(xrange(len(vecs)),3)
        xvec = vecs[x]
        avec = vecs[a]
        bvec = vecs[b]
        xdoc = random.choice(vecs_to_docs[xvec])
        adoc = random.choice(vecs_to_docs[avec])
        bdoc = random.choice(vecs_to_docs[bvec])
        #xvec = docs_to_vecs[docs[x]]
        #avec = docs_to_vecs[docs[a]]
        #bvec = docs_to_vecs[docs[b]]
        #xvec = [vec for (vec, ds) in vecs_to_docs.items() if docs[x] in ds][0]
        #avec = [vec for (vec, ds) in vecs_to_docs.items() if docs[a] in ds][0]
        #bvec = [vec for (vec, ds) in vecs_to_docs.items() if docs[b] in ds][0]
        xArray = np.array(xvec)
        aArray = np.array(avec)
        bArray = np.array(bvec)
        # a_dist = numpy.linalg.norm(xArray-aArray)
        # b_dist = numpy.linalg.norm(xArray-bArray)
        a_dist = np.linalg.norm(xArray-aArray)
        b_dist = np.linalg.norm(xArray-bArray)
        good = (a_dist > 3 or b_dist > 3) and (a_dist < 2 or b_dist < 2)
        if a_dist == b_dist or not good:
            continue
        if b_dist > a_dist:
            question = (xdoc,adoc,bdoc,a_dist,b_dist)
        else:
            question = (xdoc,bdoc,adoc,b_dist,a_dist)
        question_list.append(question)
        i += 1

    return question_list
