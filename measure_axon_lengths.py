import argparse
import pickle
import os
from tqdm import tqdm

from core.Skeleton import Skeleton
from core.AxonModel import AxonModel
from core.Branch import BranchSeq

def main(axons_path, trial_path, branches_path, original, shuffle_types, score_dict_path, num_trials, offset, verbose):
    with open(axons_path, 'rb') as f:
        axons_dict = pickle.load(f)

    with open(branches_path, 'rb') as f:
        all_branches = pickle.load(f)

    if score_dict_path:
        with open(score_dict_path, 'rb') as f:
            score_dict = pickle.load(f)

    if verbose:
        print('Finished loading data.')

    if original:
        real_axon_lengths = [AxonModel.get_total_length(mst)/1000 for mst in tqdm(axons_dict.values())]

        if verbose:
            print('Finished calculating real axon lengths.')
        
        with open(os.path.join(trial_path, 'original', 'trial_0_axonlengths.pkl'), 'wb') as f:
            pickle.dump(real_axon_lengths, f)
        if verbose:
            print('Finished saving real axon lengths.')
    
    for shuffle in shuffle_types:
        for i in range(num_trials):
            shuffled_branches = BranchSeq.get_shuffle(all_branches, shuffle, score_dict, update_position=True)
            if verbose:
                print('Finished shuffling branches.')

            new_axons_dict = {}
            for pre_cell_id in tqdm(axons_dict):
                new_axons_dict[pre_cell_id] = axons_dict[pre_cell_id].copy()

            for post_cell_id in tqdm(shuffled_branches):
                for branch in shuffled_branches[post_cell_id]:
                    for precell, syn_id, syn_pos in zip(branch.cell_id_sequence['collapsed'], branch.syn_id_sequence['collapsed'], branch.syn_pos_sequence['collapsed']):
                        new_axons_dict[precell].nodes[syn_id].update({'pos': syn_pos})

            shuffled_axon_lengths = []
            for old_mst in tqdm(new_axons_dict.values()):
                node_dict = dict(old_mst.nodes(data=True))
                soma_xyz = old_mst.nodes[-1]['pos'].squeeze()
                new_mst = Skeleton.skeleton_from_points(soma_xyz, node_dict, syn_k=8, soma_k=8)
                shuffled_axon_lengths.append(AxonModel.get_total_length(new_mst)/1000)
            
            with open(os.path.join(trial_path, shuffle, f'trial_{i+offset}_axonlengths.pkl'), 'wb') as f:
                pickle.dump(shuffled_axon_lengths, f)
            if verbose:
                print(f'Finished saving {shuffle} trial {i+offset} axon lengths.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--axon_path', type=str, required=True)
    parser.add_argument('--trial_path', type=str, required=True)
    parser.add_argument('--branches_path', type=str, default=None)
    parser.add_argument('--original', action='store_true')
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--type', action='store_true')
    parser.add_argument('--axon', action='store_true')
    parser.add_argument('--continuous', action='store_true')
    parser.add_argument('--score_dict_path', type=str, default=None)
    parser.add_argument('--num_trials', type=int, default=1)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    shuffle_types = []
    if args.random:
        shuffle_types.append('random')
    if args.type:
        shuffle_types.append('type')
    if args.axon:
        if not args.score_dict_path:
            raise ValueError('Must provide score_dict_path if axon is True')
        shuffle_types.append('axon')
    if args.continuous:
        if not args.score_dict_path:
            raise ValueError('Must provide score_dict_path if continuous is True')
        shuffle_types.append('continuous')
        
    if len(shuffle_types) > 0 and args.branches_path is None:
        raise ValueError('Must provide branches_path if a shuffle type is selected')

    main(args.axon_path, args.trial_path, 
         args.branches_path, args.original, shuffle_types, args.score_dict_path, 
         args.num_trials, args.offset, 
         args.verbose)