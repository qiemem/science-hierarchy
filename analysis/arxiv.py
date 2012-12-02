import arxiv_data.db2py as db2py
import doccluster

dd = doccluster.DendroDoc(db2py.get_docs('arxiv_data/arxiv.sqlite'))
