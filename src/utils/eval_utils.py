'''
Reference: 
https://github.com/bio-ontology-research-group/deepgo2/blob/main/deepgo/metrics.py
https://github.com/bio-ontology-research-group/deepgo2/blob/main/deepgo/utils.py

'''
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
import sys
from collections import deque
import time
import logging
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from sklearn.metrics import roc_auc_score
from scipy.spatial import distance
from scipy import sparse
import math
from collections import deque, Counter
import warnings
import pandas as pd
import numpy as np
from xml.etree import ElementTree as ET
import math
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict

BIOLOGICAL_PROCESS = 'GO:0008150'
MOLECULAR_FUNCTION = 'GO:0003674'
CELLULAR_COMPONENT = 'GO:0005575'

FUNC_DICT = {
    'cc': CELLULAR_COMPONENT,
    'mf': MOLECULAR_FUNCTION,
    'bp': BIOLOGICAL_PROCESS}

NAMESPACES = {
    'cc': 'cellular_component',
    'mf': 'molecular_function',
    'bp': 'biological_process'
}


def propagate_annots(preds, go, terms_dict):
    prop_annots = {}
    for go_id, j in terms_dict.items():
        score = preds[j]
        for sup_go in go.get_ancestors(go_id):
            if sup_go in prop_annots:
                prop_annots[sup_go] = max(prop_annots[sup_go], score)
            else:
                prop_annots[sup_go] = score
    for go_id, score in prop_annots.items():
        if go_id in terms_dict:
            preds[terms_dict[go_id]] = score
    return preds


class Ontology(object):

    def __init__(self, filename='data/go.obo', with_rels=False):
        self.ont = self.load(filename, with_rels)
        self.ic = None
        self.ic_norm = 0.0
        self.ancestors = {}
        self.compute_all_levels_and_depths()

    def has_term(self, term_id):
        return term_id in self.ont

    def get_term(self, term_id):
        if self.has_term(term_id):
            return self.ont[term_id]
        return None

    def calculate_ic(self, annots):
        cnt = Counter()
        for x in annots:
            cnt.update(x)
        self.ic = {}
        for go_id, n in cnt.items():
            parents = self.get_parents(go_id)
            if len(parents) == 0:
                min_n = n
            else:
                min_n = min([cnt[x] for x in parents])

            self.ic[go_id] = math.log(min_n / n, 2)
            self.ic_norm = max(self.ic_norm, self.ic[go_id])
    
    def get_ic(self, go_id):
        if self.ic is None:
            raise Exception('Not yet calculated')
        if go_id not in self.ic:
            return 0.0
        return self.ic[go_id]

    def get_norm_ic(self, go_id):
        return self.get_ic(go_id) / self.ic_norm

    def load(self, filename, with_rels):
        ont = dict()
        obj = None
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line == '[Term]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = dict()
                    obj['is_a'] = list()
                    obj['part_of'] = list()
                    obj['regulates'] = list()
                    obj['alt_ids'] = list()
                    obj['is_obsolete'] = False
                    continue
                elif line == '[Typedef]':
                    if obj is not None:
                        ont[obj['id']] = obj
                    obj = None
                else:
                    if obj is None:
                        continue
                    l = line.split(": ")
                    if l[0] == 'id':
                        obj['id'] = l[1]
                    elif l[0] == 'alt_id':
                        obj['alt_ids'].append(l[1])
                    elif l[0] == 'namespace':
                        obj['namespace'] = l[1]
                    elif l[0] == 'is_a':
                        obj['is_a'].append(l[1].split(' ! ')[0])
                    elif with_rels and l[0] == 'relationship':
                        it = l[1].split()
                        # add all types of relationships
                        obj['is_a'].append(it[1])
                    elif l[0] == 'name':
                        obj['name'] = l[1]
                    elif l[0] == 'is_obsolete' and l[1] == 'true':
                        obj['is_obsolete'] = True
            if obj is not None:
                ont[obj['id']] = obj
        for term_id in list(ont.keys()):
            for t_id in ont[term_id]['alt_ids']:
                ont[t_id] = ont[term_id]
            if ont[term_id]['is_obsolete']:
                del ont[term_id]
        for term_id, val in ont.items():
            if 'children' not in val:
                val['children'] = set()
            for p_id in val['is_a']:
                if p_id in ont:
                    if 'children' not in ont[p_id]:
                        ont[p_id]['children'] = set()
                    ont[p_id]['children'].add(term_id)
     
        return ont

    def get_ancestors(self, term_id):
        if term_id not in self.ont:
            return set()
        if term_id in self.ancestors:
            return self.ancestors[term_id]
        term_set = set()
        q = deque()
        q.append(term_id)
        while(len(q) > 0):
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for parent_id in self.ont[t_id]['is_a']:
                    if parent_id in self.ont:
                        q.append(parent_id)
        self.ancestors[term_id] = term_set
        return term_set

    def get_prop_terms(self, terms):
        prop_terms = set()
        for term_id in terms:
            prop_terms |= self.get_ancestors(term_id)
        return prop_terms


    def get_parents(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        for parent_id in self.ont[term_id]['is_a']:
            if parent_id in self.ont:
                term_set.add(parent_id)
        return term_set


    def get_namespace_terms(self, namespace):
        terms = set()
        for go_id, obj in self.ont.items():
            if obj['namespace'] == namespace:
                terms.add(go_id)
        return terms

    def get_namespace(self, term_id):
        return self.ont[term_id]['namespace']
    
    def get_term_set(self, term_id):
        if term_id not in self.ont:
            return set()
        term_set = set()
        q = deque()
        q.append(term_id)
        while len(q) > 0:
            t_id = q.popleft()
            if t_id not in term_set:
                term_set.add(t_id)
                for ch_id in self.ont[t_id]['children']:
                    q.append(ch_id)
        return term_set

    def get_roots(self):
        """Get all root nodes (nodes with no parents) in the ontology."""
        roots = set()
        for term_id, term in self.ont.items():
            if not term['is_a']:  # No parents
                roots.add(term_id)
        return roots

    def compute_node_level(self, term_id):
        """
        Compute the shortest distance (minimum number of edges) from any root to the node.
        Returns -1 if term_id doesn't exist.
        """
        if term_id not in self.ont:
            return -1
            
        # Return cached value if available
        if term_id in self.levels:
            return self.levels[term_id]
            
        # For root nodes
        if not self.ont[term_id]['is_a']:
            self.levels[term_id] = 0
            return 0
            
        # Find minimum level among parents and add 1
        parent_levels = []
        for parent_id in self.ont[term_id]['is_a']:
            if parent_id in self.ont:
                parent_levels.append(self.compute_node_level(parent_id))
                
        min_parent_level = min(parent_levels) if parent_levels else -1
        level = min_parent_level + 1 if min_parent_level >= 0 else -1
        
        self.levels[term_id] = level
        return level

    def compute_node_depth(self, term_id):
        """
        Compute the longest distance (maximum number of edges) from any root to the node.
        Returns -1 if term_id doesn't exist.
        """
        if term_id not in self.ont:
            return -1
            
        # Return cached value if available
        if term_id in self.depths:
            return self.depths[term_id]
            
        # For root nodes
        if not self.ont[term_id]['is_a']:
            self.depths[term_id] = 0
            return 0
            
        # Find maximum depth among parents and add 1
        parent_depths = []
        for parent_id in self.ont[term_id]['is_a']:
            if parent_id in self.ont:
                parent_depths.append(self.compute_node_depth(parent_id))
                
        max_parent_depth = max(parent_depths) if parent_depths else -1
        depth = max_parent_depth + 1 if max_parent_depth >= 0 else -1
        
        self.depths[term_id] = depth
        return depth

    def compute_all_levels_and_depths(self):
        """
        Compute levels and depths for all nodes in the ontology.
        This is more efficient than computing them one by one.
        """
        # Clear existing caches
        self.levels = {}
        self.depths = {}
        
        # First, identify all nodes
        all_terms = set(self.ont.keys())
        
        # Compute for all nodes
        for term_id in all_terms:
            if term_id not in self.levels:
                self.compute_node_level(term_id)
            if term_id not in self.depths:
                self.compute_node_depth(term_id)

    def get_level(self, term_id):
        """Get the level of a node. Computes it if not already cached."""
        if term_id not in self.levels:
            return self.compute_node_level(term_id)
        return self.levels[term_id]

    def get_depth(self, term_id):
        """Get the depth of a node. Computes it if not already cached."""
        if term_id not in self.depths:
            return self.compute_node_depth(term_id)
        return self.depths[term_id]

def compute_metrics(test_df, go, terms_dict, terms, ont, eval_preds):
    labels = np.zeros((len(test_df), len(terms_dict)), dtype=np.float32)
    eval_preds = torch.sigmoid(torch.Tensor(eval_preds)).numpy()
    for i, row in enumerate(test_df.itertuples()):
        # print(row)
        for go_id in row.prop_annotations:
            if go_id in terms_dict:
                labels[i, terms_dict[go_id]] = 1
    
    # print(labels.shape, eval_preds.shape)
    total_n = 0
    total_sum = 0
    for go_id, i in terms_dict.items():
        pos_n = np.sum(labels[:, i])
        if pos_n > 0 and pos_n < len(test_df):
            total_n += 1
            roc_auc  = compute_roc(labels[:, i], eval_preds[:, i])
            total_sum += roc_auc

    avg_auc = total_sum / total_n # macro-AUROC 
    
    print('Computing Fmax')
    fmax = 0.0
    tmax = 0.0
    wfmax = 0.0
    wtmax = 0.0
    avgic = 0.0
    precisions = []
    recalls = []
    smin = 1000000.0
    rus = []
    mis = []
    go_set = go.get_namespace_terms(NAMESPACES[ont])
    go_set.remove(FUNC_DICT[ont])
    labels = test_df['prop_annotations'].values
    labels = list(map(lambda x: set(filter(lambda y: y in go_set, x)), labels))
    spec_labels = test_df['exp_annotations'].values
    spec_labels = list(map(lambda x: set(filter(lambda y: y in go_set, x)), spec_labels))
    fmax_spec_match = 0
    for t in range(0, 101):
        threshold = t / 100.0
        preds = [set() for _ in range(len(test_df))]
        for i in range(len(test_df)):
            annots = set()
            #print(threshold, eval_preds[i])
            above_threshold = np.argwhere(eval_preds[i] >= threshold).flatten()
            for j in above_threshold:
                annots.add(terms[j])
        
            if t == 0:
                preds[i] = annots
                continue
            preds[i] = annots
        # Filter classes
        preds = list(map(lambda x: set(filter(lambda y: y in go_set, x)), preds))
        #print(labels, preds)
        fscore, prec, rec, s, ru, mi, fps, fns, avg_ic, wf = evaluate_annotations(go, labels, preds)
        spec_match = 0
        for i, row in enumerate(test_df.itertuples()):
            spec_match += len(spec_labels[i].intersection(preds[i]))
        precisions.append(prec)
        recalls.append(rec)
        if fmax < fscore:
            fmax = fscore
            tmax = threshold
            avgic = avg_ic
            fmax_spec_match = spec_match
        if wfmax < wf:
            wfmax = wf
            wtmax = threshold
        if smin > s:
            smin = s
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_index = np.argsort(recalls)
    recalls = recalls[sorted_index]
    precisions = precisions[sorted_index]
    aupr = np.trapz(precisions, recalls) # protein-centric AUPR
    

    return fmax, smin, tmax, wfmax, wtmax, avg_auc, aupr, avgic, fmax_spec_match


def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class, micro-avering 
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)
    # Compute macro-averaged ROC
    # roc_auc = roc_auc_score(labels, preds, average="macro", multi_class="ovr")
    return roc_auc

def compute_mcc(labels, preds):
    # Computes MCC for each class
    mcc = matthews_corrcoef(labels.flatten(), preds.flatten())
    return mcc

def evaluate_annotations(go, real_annots, pred_annots):
    """
    Computes Fmax, Smin, WFmax and Average IC
    Args:
       go (utils.Ontology): Ontology class instance with go.obo
       real_annots (set): Set of real GO classes
       pred_annots (set): Set of predicted GO classes
    """
    total = 0
    p = 0.0
    r = 0.0
    wp = 0.0
    wr = 0.0
    p_total= 0
    ru = 0.0
    mi = 0.0
    avg_ic = 0.0
    fps = []
    fns = []
    for i in range(len(real_annots)):
        if len(real_annots[i]) == 0:
            continue
        # print(real_annots[i], pred_annots[i])
        tp = set(real_annots[i]).intersection(set(pred_annots[i]))
        fp = pred_annots[i] - tp
        fn = real_annots[i] - tp
        tpic = 0.0
        for go_id in tp:
            tpic += go.get_norm_ic(go_id)
            avg_ic += go.get_ic(go_id)
        fpic = 0.0
        for go_id in fp:
            fpic += go.get_norm_ic(go_id)
            mi += go.get_ic(go_id)
        fnic = 0.0
        for go_id in fn:
            fnic += go.get_norm_ic(go_id)
            ru += go.get_ic(go_id)
        fps.append(fp)
        fns.append(fn)
        tpn = len(tp)
        fpn = len(fp)
        fnn = len(fn)
        total += 1
        recall = tpn / (1.0 * (tpn + fnn))
        r += recall
        wrecall = tpic / (tpic + fnic)
        wr += wrecall
        if len(pred_annots[i]) > 0:
            p_total += 1
            precision = tpn / (1.0 * (tpn + fpn))
            p += precision
            if tpic + fpic > 0:
                wp += tpic / (tpic + fpic)
    avg_ic = (avg_ic + mi) / total
    ru /= total
    mi /= total
    r /= total
    wr /= total
    if p_total > 0:
        p /= p_total
        wp /= p_total
    f = 0.0
    wf = 0.0
    if p + r > 0:
        f = 2 * p * r / (p + r)
        wf = 2 * wp * wr / (wp + wr)
    s = math.sqrt(ru * ru + mi * mi)
    return f, p, r, s, ru, mi, fps, fns, avg_ic, wf


from sklearn.metrics import auc, precision_recall_curve

def compute_macro_aupr(gt, pred):
    """
    Computes macro-AUPR for multi-label classification.

    Args:
    - gt: Ground truth binary matrix of shape (N, C), where N is the number of samples and C is the number of classes.
    - pred: Predicted scores matrix of shape (N, C).

    Returns:
    - macro_aupr: Macro-AUPR value.
    """
    # Ensure inputs are numpy arrays
    gt = np.array(gt)
    pred = np.array(pred)

    num_classes = gt.shape[1]
    aupr_values = []

    # Compute AUPR for each class
    for i in range(num_classes):
        if np.sum(gt[:, i]) > 0:  # Check if the class has at least one positive sample
            precision, recall, _ = precision_recall_curve(gt[:, i], pred[:, i])
            aupr = auc(recall, precision)  # Compute AUPR using trapezoidal rule
            aupr_values.append(aupr)
        else:
            # If no positive samples for the class, append 0 (or use np.nan for ignoring)
            aupr_values.append(0)

    # Compute macro-AUPR
    macro_aupr = np.mean(aupr_values)

    return macro_aupr, aupr_values
def stratify_aupr_values(aupr_values, go_terms, go, threshold=4):
    """
    Stratifies pre-computed AUPR values based on GO term depth and level.
    
    Args:
        aupr_values: List of AUPR values for each GO term
        go_terms: List of GO term IDs
        go: GOGraph object
        threshold: Depth/level threshold for stratification (default: 4)
    
    Returns:
        dict: Dictionary containing stratified results
    """
    # Get depths and levels
    depths = [go.get_depth(term) for term in go_terms]
    levels = [go.get_level(term) for term in go_terms]
    
    # Initialize lists for each category
    shallow_depth_auprs = []
    deep_depth_auprs = []
    shallow_level_auprs = []
    deep_level_auprs = []
    
    # Stratify based on depth and level
    for i, aupr in enumerate(aupr_values):
        # Stratify by depth
        if depths[i] <= threshold:
            shallow_depth_auprs.append(aupr)
        else:
            deep_depth_auprs.append(aupr)
            
        # Stratify by level
        if levels[i] <= threshold:
            shallow_level_auprs.append(aupr)
        else:
            deep_level_auprs.append(aupr)
    
    # Compute statistics
    results = {
        'depth': {
            'shallow': {
                'macro_aupr': np.mean(shallow_depth_auprs),
                'std': np.std(shallow_depth_auprs),
                'count': len(shallow_depth_auprs)
            },
            'deep': {
                'macro_aupr': np.mean(deep_depth_auprs),
                'std': np.std(deep_depth_auprs),
                'count': len(deep_depth_auprs)
            }
        },
        'level': {
            'shallow': {
                'macro_aupr': np.mean(shallow_level_auprs),
                'std': np.std(shallow_level_auprs),
                'count': len(shallow_level_auprs)
            },
            'deep': {
                'macro_aupr': np.mean(deep_level_auprs),
                'std': np.std(deep_level_auprs),
                'count': len(deep_level_auprs)
            }
        }
    }
    
    return results

def save_stratified_results(results, save_path):
    """
    Saves the stratified results to TSV files.
    
    Args:
        results: Dictionary containing stratified results
        save_path: Path object where files should be saved
    """
    # Save depth-based results
    depth_file = save_path / 'evaluation_depth_stratified.tsv'
    with open(depth_file, 'w') as f:
        f.write("Category\tCount\tMacro_AUPR\tStd_Dev\n")
        for category in ['shallow', 'deep']:
            res = results['depth'][category]
            f.write(f"{category}\t{res['count']}\t{res['macro_aupr']:.4f}\t{res['std']:.4f}\n")
    
    # Save level-based results
    level_file = save_path / 'evaluation_level_stratified.tsv'
    with open(level_file, 'w') as f:
        f.write("Category\tCount\tMacro_AUPR\tStd_Dev\n")
        for category in ['shallow', 'deep']:
            res = results['level'][category]
            f.write(f"{category}\t{res['count']}\t{res['macro_aupr']:.4f}\t{res['std']:.4f}\n")


def print_stratified_results(results):
    """
    Prints the stratified results in a readable format.
    
    Args:
        results: Dictionary containing stratified results
    """
    print("\nResults stratified by depth (threshold = 4):")
    print(f"{'Category':<10} {'Count':<8} {'Macro AUPR':<12} {'Std Dev':<10}")
    print("-" * 40)
    for category in ['shallow', 'deep']:
        res = results['depth'][category]
        print(f"{category:<10} {res['count']:<8d} {res['macro_aupr']:.4f}     {res['std']:.4f}")
    
    print("\nResults stratified by level (threshold = 4):")
    print(f"{'Category':<10} {'Count':<8} {'Macro AUPR':<12} {'Std Dev':<10}")
    print("-" * 40)
    for category in ['shallow', 'deep']:
        res = results['level'][category]
        print(f"{category:<10} {res['count']:<8d} {res['macro_aupr']:.4f}     {res['std']:.4f}")

