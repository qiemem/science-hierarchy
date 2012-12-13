import arxiv_data.db2py as db2py
import doccluster as dc
import sys

docs = list(set(db2py.get_docs('arxiv_data/arxiv.sqlite')))
vecs_to_docs, keywords = dc.vectorize_corpus(docs)

if sys.argv[1] == 'cluster':
    import clustertestmaker
    tree = dc.make_tree(vecs_to_docs)
    qs = clustertestmaker.make_test(tree, vecs_to_docs)
    choice_type = 3
    dc.write_tree('qcluster.tree', tree, keywords)
else:
    import testmaker
    qs = testmaker.make_dist_test(docs,vecs_to_docs,50)
    choice_type = 1
with open('keywords.txt','w') as out:
    out.write('\n'.join(keywords))
with open('qs.txt','w') as out:
	out.write('\n'.join(map(str,qs)))
import json

def to_dict(pk, prompt, test, controlChoice):
	return {
		'pk': pk,
		'model': 'packages.question', 
		'fields' : {
			'prompt': '\n'.join(prompt),
			'test': '\n'.join(test),
			'controlChoice': '\n'.join(controlChoice),
			'selected': -1,
			'choiceType': choice_type
		}
	}

q_dicts = [to_dict(i+1, p, t, c) for (i, (p,t,c,_,_)) in enumerate(qs)]
print(json.dumps(q_dicts))
