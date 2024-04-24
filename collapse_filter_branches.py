import argparse
import pickle
import os
import json
from collections import defaultdict
from tqdm import tqdm

from core.GenericBranch import GenericBranchSeq

def main(branches_path, output_path, filter_dict, min_intersection=3, verbose=False):
    
    with open(branches_path, 'rb') as f:
        all_branches = pickle.load(f)
    
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

    branches_path = '/drive_sdc/ssarup/flywire_data/mushroom_body/mb_branches.pkl'
    output_path = '/drive_sdc/ssarup/flywire_data/mushroom_body/dopamine'
    min_intersection = 3
    verbose = True

    filter_dict = {
        'top_nts': ['dopamine'],
    }

    main(branches_path, output_path, filter_dict,
         min_intersection=min_intersection, 
         verbose=verbose)