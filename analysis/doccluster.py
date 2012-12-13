from collections import Counter, defaultdict
import random
import neuro_data.db2py as db
import tfidf.tfidf as tfidf
import kmeans
import numpy as np
import os
import re


def vectorize_corpus(docs, n_keywords = 100, n_doc_keywords = 3, tfidf_vecs = False):
    """
    Returns a dictionary of vectors mapped to their corresponding documents as
    well as the keywords used. Uses reasonable defaults, but lets you control
    major parameters. USE THIS.
    
    Inputs:
    docs - A list of (title, abstract) pairs
    n_keywords - Number of keywords to use, ie the length of the resulting vectors
        Defaults to 100.
    n_doc_keywords - Number of top words to use from each document in creating
        the list of keywords. Defualts to 3.
    tfidf_vecs - If True, tfidf values of each word are used as elements of the
        vectors. Otherwise, uses word containment. Defaults to False.

    Output:
    vecs_to_docs - A dict from vectors to lists of documents.
    keywords - The list of keywords used.
    """
    clean_docs = [clean_doc(doc) for doc in docs]
    idf = get_idf(clean_docs)
    keywords = get_corpus_keywords(clean_docs, idf, n_keywords, n_doc_keywords)
    if tfidf_vecs:
        def doc_quant(doc):
            doc_dict = dict(idf.get_doc_keywords(doc))
            return lambda w : doc_dict.get(word, 0)
    else:
        def doc_quant(doc):
            doc_words = set(doc.split())
            return lambda w : w in doc_words
    vecs_to_docs = get_vecs_to_docs(clean_docs, keywords, doc_quant)
    clean_to_raw = dict(zip(clean_docs, docs))
    vecs_to_docs = {k : [clean_to_raw[d] for d in v] for (k,v) in vecs_to_docs.items()}
    return vecs_to_docs, keywords

def make_tree(vecs_to_docs):
    """
    Clusters a vector->docs dict.
    """
    points = np.array(list(vecs_to_docs.keys()))
    return kmeans.bisecting_kmeans(points)

def clean_doc(doc):
    """
    Input:
        doc - (title, body) pair
    Output:
        The title and body are concatenated and stripped of non alphnum and 
        lower cased.
    """
    title, text = doc
    return strip_non_alphanum(title+' '+text).lower()

def strip_non_alphanum(string, pattern=re.compile('[^A-Za-z ]')):
    """
    Strips everything but letters, numbers, and spaces.
    """
    return pattern.sub('', ' '.join(string.split()))

def get_idf(clean_docs, stopword_ratio = .15):
    idf = tfidf.TfIdf()
    for doc in clean_docs:
        idf.add_input_document(doc)
    idf.stopwords = idf.calculate_stopwords(stopword_ratio)
    return idf

def get_corpus_keywords(clean_docs, idf = None, n_keywords = 100, n_doc_keywords = 3,
        min_word_length = 3):
    if not idf:
        idf = get_idf(clean_docs)
    top_words = [(word,score) for doc in clean_docs 
            for (word,score) in idf.get_doc_keywords(doc)[:n_doc_keywords]
            if len(word)>=min_word_length and score>0]
    top_word_counts = Counter(p[0] for p in top_words)
    keywords = [word for (word, _) in top_word_counts.most_common(n_keywords)]
    return keywords

def vectorize(keywords, word_quant):
    return tuple(word_quant(word) for word in keywords)

def containment_vec(doc):
    return lambda w : w in doc

def get_vecs_to_docs(clean_docs, keywords, 
        doc_quant = lambda d : (lambda w : w in d)):
    vecs_to_docs = defaultdict(list)
    for doc in clean_docs:
        word_quant = doc_quant(doc)
        vecs_to_docs[vectorize(keywords, word_quant)].append(doc)
    return vecs_to_docs

def word_value_pairs(keywords, vector):
    """
    Converts returns a list of (word, value) pairs for the given vector,
    sorted by value.
    """
    word_value_pairs = zip(keywords, vector)
    return sorted(word_value_pairs, key=lambda p:abs(p[1]), reverse=True)

class DendroDoc:
    def __init__(self, docs, num_vec_words=100, num_doc_keywords=3, stopword_ratio=.15,
            min_word_length=3):
        """
        Calculates idf values. words that appear in more than `stopword_ratio`
        of the documents are given an idf of 0. Gets top `num_doc_keywords` from
        each doc and picks the `num_vec_words` most common.
        
        Input:
        docs - A list of (title, body) pairs.
        num_vec_words - Number of words to make the feature space out of. By
            definition, the number of dimensions. Defaults to 100.
        num_doc_keywords - Number of keywords to pick from each document.
            Defaults to 3.
        """
        self.docs = docs
        self.clean_docs = [clean_doc(doc) for doc in self.docs]
        self.idf = tfidf.TfIdf()
        for doc in self.clean_docs:
            self.idf.add_input_document(doc)
        self.idf.stopwords = self.idf.calculate_stopwords(.15) # .15 just based on trying random nums

        # At > top 3, I started seeing stopwords enter
        self.top_words = [(word,score) for doc in self.clean_docs 
                for (word,score) in self.idf.get_doc_keywords(doc)[:num_doc_keywords]
                if len(word)>=min_word_length and score>0]
        self.top_word_counts = Counter(p[0] for p in self.top_words)
        self.key_words = [word for (word, _) in self.top_word_counts.most_common(num_vec_words)]

        self.vecs_to_docs = defaultdict(list)
        for (raw_doc, doc) in zip(self.docs, self.clean_docs):
            doc_dict = dict(self.idf.get_doc_keywords(doc))
            vector = tuple(doc_dict.get(word, 0) for word in self.key_words)
            self.vecs_to_docs[vector].append(raw_doc)
        self.doc_vectors = np.array(list(self.vecs_to_docs.keys()))
        self.tree = kmeans.bisecting_kmeans(self.doc_vectors)

    def get_docs(self, vector):
        """
        Retrieves the list of docs for the given vector.
        """
        return self.vecs_to_docs[tuple(vector)]

    def word_value_pairs(self, vector):
        """
        Converts returns a list of (word, value) pairs for the given vector,
        sorted by value.
        """
        word_value_pairs = zip(self.key_words, vector)
        return sorted(word_value_pairs, key=lambda p:abs(p[1]), reverse=True)

    def word_diffs(self,tree):
        """
        Shows what the tree is splitting on. A positive value means the left
        child features that word more.
        """
        return self.word_value_pairs(tree.children[0].mean - tree.children[1].mean)

    def write_gdf(self, filename, tree=None):
        """
        Writes a gdf file for the tree. Displayable with gephi.
        """
        if tree is None:
            tree = self.tree
        with open(filename,'w') as f:
            i = 0
            tree_ids = {}
            f.write('nodedef>name VARCHAR,words VARCHAR\n')
            for t in tree.dft():
                tree_ids[t]=i
                if hasattr(t, 'means'):
                    words = ' '.join(':'.join(map(str,p)) for p in self.word_diffs(t)[:5])
                else:
                    words = 'leaf'
                f.write('{},{}\n'.format(i, words))
                i+=1
            f.write('edgedef>node1 VARCHAR,node2 VARCHAR\n')
            for t in tree.dft():
                try:
                    c1,c2 = t.children
                    f.write('{},{}\n'.format(tree_ids[t], tree_ids[c1]))
                    f.write('{},{}\n'.format(tree_ids[t], tree_ids[c2]))
                except AttributeError:
                    pass

def write_spato(filename, tree, vecs_to_docs, keywords, depth = 23):
    """
    Writes a spato file for the tree. Displayable with SPaTo.
    """
    if os.path.isfile(filename):
        os.remove(filename)
    if not os.path.exists(filename):
        os.makedirs(filename)
    with open(os.path.join(filename, 'document.xml'),'w') as docfile:
        docfile.write('''<?xml version="1.0" ?>
<document>
    <title>Document Dendrogram</title>
    <description>blah blah blah</description>
    <nodes src="nodes.xml" />
    <links src="links.xml" />
</document>
''')
    tree_ids = {}
    with open(os.path.join(filename, 'nodes.xml'),'w') as nodesfile:
        nodesfile.write('<?xml version="1.0 ?>\n<nodes>\n  <projection name="LonLat" />\n')
        i = 1
        for t in tree.dft(depth):
            if len(t.children)>0:
                words = ' '.join(':'.join(map(str, p)) for p in word_value_pairs(keywords,t.mean)[:5])
            else:
                vec = t.mean
                words = '|'.join(strip_non_alphanum(title) for (title, _) in vecs_to_docs[tuple(vec)])
            nodesfile.write('  <node id="id{}" name="{}" location="{},{}" strength="1" />\n'.format(i, words, np.random.random(), np.random.random()))
            tree_ids[t] = i
            i+=1
        nodesfile.write('</nodes>')
    with open(os.path.join(filename, 'links.xml'),'w') as linksfile:
        linksfile.write('<?xml version="1.0" ?>\n<links inverse="false">\n')
        parents = {}
        for t in tree.dft(depth):
            parent = tree_ids[t]
            linksfile.write('  <source index="{}">\n'.format(parent))
            if hasattr(t, 'children'):
                for c in t.children:
                    if c in tree_ids:
                        cid = tree_ids[c]
                        parents[cid] = parent
                        linksfile.write('    <target index="{}" weight="1" />\n'.format(cid))
            if parent in parents:
                linksfile.write('    <target index="{}" weight="1" />\n'.format(parents[parent]))
            linksfile.write('  </source>\n')
        linksfile.write('</links>')

def write_treeviz(filename, tree, vecs_to_docs, keywords, depth = -1):
    with open(filename, 'w') as out:
        def write_node(t, d):
            if len(t.children)==0:
                name = strip_non_alphanum(vecs_to_docs[tuple(t.mean)][0][0])
            else:
                name = ' '.join(w for (w,v) in word_value_pairs(keywords,t.mean)[:10] if v>0)
            out.write('<L{} name="{}" weight="1" depth="{}">\n'.format(d, name, d))
            if d != depth:
                for c in t.children:
                    write_node(c,d+1)
            out.write('</L{}>\n'.format(d))
        write_node(tree, 1)

def write_tree(filename, tree, keywords):
    with open(filename, 'w') as out:
        def write_node(t, d):
            out.write('{}{}\n'.format(' '*d, ','.join(str(x) for x in t.mean)))
            for c in t.children:
                write_node(c, d+1)
        out.write(','.join(keywords)+'\n')
        write_node(tree, 0)

def write_json(filename, tree, vecs_to_docs, keywords, depth=-1):
    import json
    colors = ['#880000', '#008800', '#000088', '#888800', '#880088', '#008888']
    with open(filename,'w') as out:
        def build_json(t, d):
            if len(t.children)==0:
                name = vecs_to_docs[tuple(t.mean)][0][0]
            else:
                name = ' '.join(w for (w,v) in word_value_pairs(keywords,t.mean)[:10] if v>0)
            return {
                'id': str(d)+') '+name,
                'name': name,
                'data': {
                    '$color': random.choice(colors)
                },
                'children': [build_json(c, d+1) for c in t.children if d!=depth]
            }
        json.dump(build_json(tree, 0), out)
