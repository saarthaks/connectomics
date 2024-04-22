import pickle
import pandas
import numpy as np
import os
from tqdm import tqdm

from core.Branch import BranchSeq
from core.GenericBranch import GenericBranchSeq
from core.Tree import Tree

def load_tree(neuron_id, datapath='/drive_sdc/ssarup/flywire_data/all_trees'):
    try:
        with open(os.path.join(datapath, f'{neuron_id}.pkl'), 'rb') as f:
            tree = pickle.load(f)
            return tree
    except:
        print(f'Error loading {neuron_id}')
        return None    

def load_tree_dict(neuron_ids, datapath='/drive_sdc/ssarup/flywire_data/all_trees'):
    neuron_ids = list(set(neuron_ids))

    trees = {}
    for neuron_id in tqdm(neuron_ids):
        tree = load_tree(neuron_id, datapath=datapath)
        if tree is not None:
            trees[neuron_id] = tree
    return trees

def load_branches(neuron_id, datapath='/drive_sdc/ssarup/flywire_data/all_branches'):
    try:
        with open(os.path.join(datapath, f'{neuron_id}.pkl'), 'rb') as f:
            branches = pickle.load(f)
            return branches
    except:
        print(f'Error loading {neuron_id}')
        return None

def load_branches_dict(neuron_ids, datapath='/drive_sdc/ssarup/flywire_data/all_branches'):
    neuron_ids = list(set(neuron_ids))

    branches = {}
    for neuron_id in tqdm(neuron_ids):
        branch = load_branches(neuron_id, datapath=datapath)
        if branch is not None:
            branches[neuron_id] = branch
    return branches
