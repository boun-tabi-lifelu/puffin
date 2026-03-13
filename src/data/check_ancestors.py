from src.utils.eval_utils import Ontology
import pandas as pd 

ont = Ontology('data/go-basic-2020-06-01.obo')

MF_terms = pd.read_csv('data/GeneOntology/MF_mapping.csv', header=None)
BP_terms = pd.read_csv('data/GeneOntology/BP_mapping.csv', header=None)
CC_terms = pd.read_csv('data/GeneOntology/CC_mapping.csv', header=None)

for subont in [MF_terms[0].tolist(), BP_terms[0].tolist(), CC_terms[0].tolist()]: 
    ancestors = set()
    for term in subont:
        term_ancestors = ont.get_ancestors(term)
        assert len(term_ancestors) > 1
        ancestors.update(term_ancestors)
    print(ancestors.difference(set(subont)))
    
