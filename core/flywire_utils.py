import pickle
import pandas
import numpy as np
import os
from tqdm import tqdm

from core.Branch import BranchSeq
from core.GenericBranch import GenericBranchSeq
from core.Tree import Tree

def load_tree(neuron_id, datapath='/drive_sdc/ssarup/flywire_data/all_trees'):
    with open(os.path.join(datapath, f'{neuron_id}.pkl'), 'rb') as f:
        tree = pickle.load(f)
    return tree
    

def load_tree_dict(neuron_ids, datapath='/drive_sdc/ssarup/flywire_data/all_trees'):
    neuron_ids = list(set(neuron_ids))

    trees = {}
    for neuron_id in tqdm(neuron_ids):
        trees[neuron_id] = load_tree(neuron_id, datapath=datapath)
    return trees

def load_branches(neuron_id, datapath='/drive_sdc/ssarup/flywire_data/all_branches'):
    with open(os.path.join(datapath, f'{neuron_id}.pkl'), 'rb') as f:
        branches = pickle.load(f)
    return branches

def load_branches_dict(neuron_ids, datapath='/drive_sdc/ssarup/flywire_data/all_branches'):
    neuron_ids = list(set(neuron_ids))

    branches = {}
    for neuron_id in tqdm(neuron_ids):
        branches[neuron_id] = load_branches(neuron_id, datapath=datapath)
    return branches
