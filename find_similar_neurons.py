import argparse
import pickle
import os
from collections import defaultdict
from tqdm import tqdm

from core.Branch import BranchSeq

def main(branch_path, output_path, min_intersection=1, verbose=False):
    
    with open(branch_path, 'rb') as f:
        all_branches = pickle.load(f)
    
    if verbose:
        print('Done loading branches.')
    
    similarity_dict = defaultdict(list)

    cell_ids = sorted(list(all_branches.keys()))
    for i in tqdm(range(len(cell_ids))):
        branches = all_branches[cell_ids[i]]
        branch_inp_cells_1 = [set(branch.cell_id_sequence['collapsed']) for branch in branches]

        for j in range(i+1, len(cell_ids)):
            other_branches = all_branches[cell_ids[j]]
            branch_inp_cells_2 = [set(branch.cell_id_sequence['collapsed']) for branch in other_branches]

            for branch_id_1, inp_cells_1 in enumerate(branch_inp_cells_1):
                for branch_id_2, inp_cells_2 in enumerate(branch_inp_cells_2):
                    if len(inp_cells_1.intersection(inp_cells_2)) >= min_intersection:
                        similarity_dict[cell_ids[i]].append((cell_ids[j], branch_id_1, branch_id_2))
    
    with open(output_path, 'wb') as f:
        pickle.dump(similarity_dict, f)
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--branches_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--min_intersection', type=int, default=3)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    main(args.branches_path, args.output_path, 
         min_intersection=args.min_intersection, 
         verbose=args.verbose)