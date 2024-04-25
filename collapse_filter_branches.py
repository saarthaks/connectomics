import argparse
import pickle
import os
import json
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

from core.flywire_utils import *
from core.GenericBranch import GenericBranchSeq

def main(all_branches, output_path, filter_dict, min_intersection=3, verbose=False):
    
    if verbose:
        print('Done loading branches.')

    with open(os.path.join(output_path, 'filter_dict.json'), 'w') as f:
        json.dump(filter_dict, f)
    
    for cell_id, branches in tqdm(all_branches.items()):
        for branch in branches:
            branch.collapse_path(filter_dict)
    
    with open(os.path.join(output_path, 'collapsed_branches.pkl'), 'wb') as f:
        pickle.dump(all_branches, f)
    if verbose:
        print('Saved collapsed branches.')
    
    similarity_dict = defaultdict(list)

    cell_ids = sorted(list(all_branches.keys()))
    for i in tqdm(range(len(cell_ids))):
        branches = all_branches[cell_ids[i]]
        branch_inp_cells_1 = [branch.collapsed_cell_id_set for branch in branches]

        for j in range(i+1, len(cell_ids)):
            other_branches = all_branches[cell_ids[j]]
            branch_inp_cells_2 = [branch.collapsed_cell_id_set for branch in other_branches]

            for branch_id_1, inp_cells_1 in enumerate(branch_inp_cells_1):
                if len(inp_cells_1) < min_intersection:
                    continue
                for branch_id_2, inp_cells_2 in enumerate(branch_inp_cells_2):
                    if len(inp_cells_2) < min_intersection:
                        continue
                    if len(inp_cells_1.intersection(inp_cells_2)) >= min_intersection:
                        similarity_dict[cell_ids[i]].append((cell_ids[j], branch_id_1, branch_id_2))
    
    with open(os.path.join(output_path, 'similarity_dict.pkl'), 'wb') as f:
        pickle.dump(similarity_dict, f)
        
if __name__ == '__main__':

    neuron_annotation = pd.read_csv('./neuron_annotation.tsv', sep='\t')

    base = '/drive_sdc/ssarup/flywire_data'
    region = 'olfactory'
    os.makedirs(os.path.join(base, region), exist_ok=True)

    cell_ids = neuron_annotation[neuron_annotation['cell_class'] == 'olfactory']['root_id'].values
    branches = load_branches_dict(cell_ids)
    pre_ids = get_pre_ids(branches)
    pre_ids_df = neuron_annotation[neuron_annotation['root_id'].isin(pre_ids)]
    pre_ids_df.to_csv(os.path.join(base, region, 'pre_ids.csv'), index=False)

    experiment_name = 'dopamine'
    filter_dict = {
        'top_nts': ['dopamine'],
    }
    min_intersection = 3
    verbose = True
    output_path = os.path.join(base, region, experiment_name)
    os.makedirs(output_path, exist_ok=True)

    main(branches, output_path, filter_dict,
         min_intersection=min_intersection, 
         verbose=verbose)