from collections import Counter, defaultdict
import neuro_data.db2py as db
import tfidf.tfidf as tfidf
import kmeans
import numpy as np
import os
import re


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
    return pattern.sub('', string)

#docs = list(set(db.get_clean_docs('neuro_data/neuro.sqlite')))
class DendroDoc:
    def __init__(self, docs, num_vec_terms=100, num_doc_keyterms=3, stopword_ratio=.15):
        """
        Calculates idf values. Terms that appear in more than `stopword_ratio`
        of the documents are given an idf of 0. Gets top `num_doc_keyterms` from
        each doc and picks the `num_vec_terms` most common.
        
        Input:
        docs - A list of (title, body) pairs.
        num_vec_terms - Number of terms to make the feature space out of. By
            definition, the number of dimensions. Defaults to 100.
        num_doc_keyterms - Number of keywords to pick from each document.
            Defaults to 3.
        """
        self.docs = docs
        self.clean_docs = [clean_doc(doc) for doc in self.docs]
        self.idf = tfidf.TfIdf()
        for doc in self.clean_docs:
            self.idf.add_input_document(doc)
        self.idf.stopterms = self.idf.calculate_stopterms(.15) # .15 just based on trying random nums

        # At > top 3, I started seeing stopwords enter
        self.top_terms = [p for doc in self.clean_docs 
                for p in self.idf.get_doc_keyterms(doc)[:num_doc_keyterms]]
        self.top_term_counts = Counter(p[0] for p in self.top_terms)
        self.key_terms = [term for (term, _) in self.top_term_counts.most_common(num_vec_terms)]

        self.vecs_to_docs = defaultdict(list)
        for (raw_doc, doc) in zip(self.docs, self.clean_docs):
            doc_dict = dict(self.idf.get_doc_keyterms(doc))
            vector = tuple(doc_dict.get(term, 0) for term in self.key_terms)
            self.vecs_to_docs[vector].append(raw_doc)
        self.doc_vectors = np.array(list(self.vecs_to_docs.keys()))
        self.tree = kmeans.bisecting_kmeans(self.doc_vectors)

    def get_docs(self, vector):
        """
        Retrieves the list of docs for the given vector.
        """
        return self.vecs_to_docs[tuple(vector)]

    def get_term_value_pairs(self, vector):
        """
        Converts returns a list of (term, value) pairs for the given vector,
        sorted by value.
        """
        term_value_pairs = zip(self.key_terms, vector)
        return sorted(term_value_pairs, key=lambda p:abs(p[1]), reverse=True)

    def term_diffs(self,tree):
        """
        Shows what the tree is splitting on. A positive value means the left
        child features that word more.
        """
        return get_term_value_pairs(tree.means[0] - tree.means[1])
        tds = zip(self.key_terms, tree.means[0]-tree.means[1])
        return sorted(tds, key=lambda p:abs(p[1]), reverse = True)


    def write_gdf(self, filename, tree=None):
        """
        Writes a gdf file for the tree. Displayable with gephi.
        """
        if tree is None:
            tree = self.tree
        with open(filename,'w') as f:
            i = 0
            tree_ids = {}
            f.write('nodedef>name VARCHAR,terms VARCHAR\n')
            for t in tree.dft():
                tree_ids[t]=i
                if hasattr(t, 'means'):
                    terms = ' '.join(':'.join(map(str,p)) for p in self.term_diffs(t)[:5])
                else:
                    terms = 'leaf'
                f.write('{},{}\n'.format(i, terms))
                i+=1
            f.write('edgedef>node1 VARCHAR,node2 VARCHAR\n')
            for t in tree.dft():
                try:
                    c1,c2 = t.children
                    f.write('{},{}\n'.format(tree_ids[t], tree_ids[c1]))
                    f.write('{},{}\n'.format(tree_ids[t], tree_ids[c2]))
                except AttributeError:
                    pass

    def write_spato(self, filename, tree=None, depth = 23):
        """
        Writes a spato file for the tree. Displayable with SPaTo.
        """
        if tree is None:
            tree = self.tree
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
                try:
                    terms = ' '.join(':'.join(map(str,p)) for p in self.term_diffs(t)[:5])
                except AttributeError:
                    vec = t.values[0]
                    terms = '|'.join(strip_non_alphanum(title) for (title, body) in self.get_docs(vec))
                nodesfile.write('  <node id="id{}" name="{}" location="{},{}" strength="1" />\n'.format(i, terms, np.random.random(), np.random.random()))
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


        


