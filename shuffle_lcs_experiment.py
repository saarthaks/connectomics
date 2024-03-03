import argparse
import pickle
import pandas as pd
import os
import concurrent.futures
from collections import defaultdict
from tqdm import tqdm

from core.Branch import BranchSeq

def get_shuffle(trial_path, trial_num, all_branches, shuffle_type='random', score_dict=None):
    if shuffle_type == 'original':
        return all_branches

    shuffled_branches = {}
    for cell_id, branches in all_branches.items():
        if shuffle_type == 'random':
            shuffled_branches[cell_id] = [branch.get_random_shuffle() for branch in branches]
        elif shuffle_type == 'type':
            shuffled_branches[cell_id] = [branch.get_type_shuffle() for branch in branches]
        elif shuffle_type == 'axon':
            if len(score_dict[cell_id]) != len(branches):
                raise ValueError('Score matrix length does not match branch length')
            shuffled_branches[cell_id] = [branch.get_axon_shuffle(score_mat) for branch, score_mat in zip(branches, score_dict[cell_id])]
        else:
            raise ValueError('Invalid shuffle type')
        
    with open(os.path.join(trial_path, shuffle_type, f'trial_{trial_num}.pkl'), 'wb') as f:
        pickle.dump(shuffled_branches, f)
    
    return shuffled_branches

def lcs_dist(similarity_dict_part, shuffled_branches, pt_root_id_to_char, reverse=False):

    all_dists = {}
    for cell_id, v in tqdm(similarity_dict_part.items()):
        all_dists[cell_id] = {}
        branch_comparisons = defaultdict(list)
        if reverse:
            for other_cell_id, branch_id_1, branch_id_2 in v:
                branch_comparisons[branch_id_1].append(shuffled_branches[other_cell_id][branch_id_2].reverse())
            
            for branch_id_1, other_branches in branch_comparisons.items():
                dists = BranchSeq.lcs_dist_list(shuffled_branches[cell_id][branch_id_1].reverse(), other_branches, pt_root_id_to_char)
                all_dists[cell_id][branch_id_1] = dists
        else:
            for other_cell_id, branch_id_1, branch_id_2 in v:
                branch_comparisons[branch_id_1].append(shuffled_branches[other_cell_id][branch_id_2])

            for branch_id_1, other_branches in branch_comparisons.items():
                dists = BranchSeq.lcs_dist_list(shuffled_branches[cell_id][branch_id_1], other_branches, pt_root_id_to_char)
                all_dists[cell_id][branch_id_1] = dists
    
    return all_dists


def main(branch_path, similarity_path, cells_path, trial_path, shuffle_type, num_trials, 
         score_dict_path=None, reverse=False, verbose=False):

    with open(branch_path, 'rb') as f:
        all_branches = pickle.load(f)
    with open(similarity_path, 'rb') as f:
        similarity_dict = pickle.load(f)
    all_cells = pd.read_csv(cells_path)
    if verbose:
        print('Done loading branches.')

    if score_dict_path:
        with open(score_dict_path, 'rb') as f:
            score_dict = pickle.load(f)
    else:
        score_dict=None

    pt_root_ids = list(all_cells['pt_root_id'].values)
    pt_root_id_to_char, char_to_pt_root_id = BranchSeq.make_id2char_dict(pt_root_ids)

    if shuffle_type == 'original':
        num_trials = 1

        if verbose:
            print('No shuffle.')
    
    for t in range(num_trials):
        if verbose:
            print(f'Starting trial {t}...')
        
        shuffled_branches = get_shuffle(trial_path, t, all_branches, shuffle_type=shuffle_type, score_dict=score_dict)
        if verbose:
            print('Finished shuffling branches.')

        all_dists = lcs_dist(similarity_dict, shuffled_branches, pt_root_id_to_char, reverse=reverse)
        if verbose:
            print('Finished calculating distances. Saving...')
        
        with open(os.path.join(trial_path, shuffle_type, f'trial_{t}_dists.pkl'), 'wb') as f:
            pickle.dump(all_dists, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--branches_path', type=str, required=True)
    parser.add_argument('--similarity_path', type=str, required=True)
    parser.add_argument('--cells_path', type=str, required=True)
    parser.add_argument('--trial_path', type=str, required=True)
    parser.add_argument('--shuffle_type', type=str, required=True)
    parser.add_argument('--score_dict_path', type=str)
    parser.add_argument('--num_trials', type=int, required=True)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    if args.shuffle_type == 'axon' and args.score_dict_path is None:
        raise ValueError('Axon shuffle requires a score matrix path')

    main(args.branches_path, args.similarity_path, args.cells_path, args.trial_path, 
         args.shuffle_type, args.num_trials,  
         score_dict_path=args.score_dict_path, reverse=args.reverse, verbose=args.verbose)
