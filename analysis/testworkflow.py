import arxiv_data.db2py as db2py
import doccluster as dc

docs = list(set(db2py.get_docs('arxiv_data/arxiv.sqlite')))
vecs_to_docs, keywords = dc.vectorize_corpus(docs)

import testmaker
qs = testmaker.make_dist_test(docs,vecs_to_docs,50)
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
			'choiceType': 1
		}
	}

q_dicts = [to_dict(i+1, p, t, c) for (i, (p,t,c,_,_)) in enumerate(qs)]
print(json.dumps(q_dicts))
