import arxiv_data.dbs_to_docs, keywords = dc.vectorize_corpus(docs) as db2py
import doccluster as dc

docs = db2py.get_docs('arxiv_data/arxiv.sqlite')
vecs_to_docs, keywords = dc.vectorize_corpus(docs)

#tree = dc.make_tree(vecs_to_docs)
