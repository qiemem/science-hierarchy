import arxiv_data.db2py as db2py
import doccluster
import tfidf.tfidf as tfidf

docs = db2py.get_docs('arxiv_data/arxiv.sqlite')
clean_docs = [doccluster.clean_doc(doc) for doc in docs]

idf = tfidf.TfIdf()
for doc in clean_docs:
    idf.add_input_document(doc)

dd = doccluster.DendroDoc(docs, 10)
