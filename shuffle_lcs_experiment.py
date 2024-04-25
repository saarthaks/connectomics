import argparse
import pickle
import pandas as pd
import os
from collections import defaultdict
from tqdm import tqdm

from core.Branch import BranchSeq
from core.GenericBranch import GenericBranchSeq
from core.plotting import get_counts

def lcs_dist(similarity_dict_part, shuffled_branches, pt_root_id_to_char):

    all_dists = {}
    for cell_id, v in tqdm(similarity_dict_part.items()):
        all_dists[cell_id] = {}
        branch_comparisons = defaultdict(list)
        for other_cell_id, branch_id_1, branch_id_2 in v:
            branch_comparisons[branch_id_1].append(shuffled_branches[other_cell_id][branch_id_2])

        for branch_id_1, other_branches in branch_comparisons.items():
            dists = BranchSeq.lcs_dist_list(shuffled_branches[cell_id][branch_id_1], other_branches, pt_root_id_to_char)
            all_dists[cell_id][branch_id_1] = dists
    
    return all_dists


def main(branch_path, similarity_path, cells_path, trial_path, shuffle_type, num_trials, offset,
         score_dict_path=None, save_shuffle=False, verbose=False):

    with open(branch_path, 'rb') as f:
        all_branches = pickle.load(f)
    with open(similarity_path, 'rb') as f:
        similarity_dict = pickle.load(f)
    all_cells = pd.read_csv(cells_path)
    if verbose:
        print('Done loading branches.')

    if score_dict_path is not None:
        with open(score_dict_path, 'rb') as f:
            score_dict = pickle.load(f)
    else:
        score_dict=None

    pt_root_ids = list(all_cells['pt_root_id'].values)
    pt_root_id_to_char, _ = BranchSeq.make_id2char_dict(pt_root_ids)

    if shuffle_type == 'original':
        num_trials = 1

        if verbose:
            print('No shuffle.')
    
    for t in range(offset, offset+num_trials):
        if verbose:
            print(f'Starting trial {t}...')
        
        shuffled_branches = GenericBranchSeq.get_shuffle(all_branches, shuffle_type=shuffle_type, score_dict=score_dict)
        if verbose:
            print('Finished shuffling branches.')
        if save_shuffle:
            with open(os.path.join(trial_path, shuffle_type, f'trial_{t}.pkl'), 'wb') as f:
                pickle.dump(shuffled_branches, f)
            if verbose:
                print('Finished saving shuffled branches.')
        
        all_dists = lcs_dist(similarity_dict, shuffled_branches, pt_root_id_to_char)
        if verbose:
            print('Finished calculating distances. Saving...')
            print('Num sequences found:', len(get_counts(all_dists)))
            print('Num unique sequences found:', len(get_counts(all_dists, unique=True)))
        
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
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--save_shuffle', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    if args.shuffle_type == 'axon' and args.score_dict_path is None:
        raise ValueError('Axon shuffle requires a score matrix path')

    main(args.branches_path, args.similarity_path, args.cells_path, args.trial_path, 
         args.shuffle_type, args.num_trials, args.offset, 
         score_dict_path=args.score_dict_path, save_shuffle=args.save_shuffle, verbose=args.verbose)
