import argparse
import pickle
import os
from collections import defaultdict
from tqdm import tqdm

from core.plotting import *

def main(trial_path, figures_path, shuffle_types, zscore=None, unique=False, dendrite=False, basal=False, transparent=False, stats=False, verbose=False):
    
    with open(os.path.join(trial_path, 'original', 'trial_0_dists.pkl'), 'rb') as f:
        all_dists = pickle.load(f)
    
    real_count = len(get_counts(all_dists, unique=unique, dendrite=dendrite, basal=basal))
    if verbose:
        print('Done counts for original distances.')
    
    all_shuffle_counts = defaultdict(list)
    for shuffle_type in shuffle_types:
        for fname in tqdm(os.listdir(os.path.join(trial_path, shuffle_type))):
            if fname.endswith('dists.pkl'):
                with open(os.path.join(trial_path, shuffle_type, fname), 'rb') as f:
                    dists = pickle.load(f)
                all_shuffle_counts[shuffle_type].append(len(get_counts(dists, unique=unique, dendrite=dendrite, basal=basal)))
        
        if verbose:
            print(f'Done loading counts for shuffle type: {shuffle_type}.')
    
    bins = 5
    fig, _ = count_histogram(real_count, all_shuffle_counts, bins=bins, zscore=zscore, dendrite=dendrite, stats=stats)

    str_comb = ''.join(shuffle_types)
    if zscore:
        str_comb += f'_z_{zscore}'
    if unique:
        str_comb += '_unique'
    if dendrite:
        str_comb += '_dendrite'
    if basal:
        str_comb += '_basal'
    fig.savefig(os.path.join(figures_path, f'lcs_histogram_{str_comb}.png'), bbox_inches='tight', transparent=transparent)
    fig.savefig(os.path.join(figures_path, f'lcs_histogram_{str_comb}.svg'), bbox_inches='tight', transparent=transparent)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial_path', type=str, required=True)
    parser.add_argument('--figures_path', type=str, required=True)
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--type', action='store_true')
    parser.add_argument('--axon', action='store_true')
    parser.add_argument('--continuous', action='store_true')
    parser.add_argument('--zscore', default=None)
    parser.add_argument('--unique', action='store_true')
    parser.add_argument('--dendrite', action='store_true')
    parser.add_argument('--basal', action='store_true')
    parser.add_argument('--transparent', action='store_true')
    parser.add_argument('--stats', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    shuffle_types = []
    if args.random:
        shuffle_types.append('random')
    if args.type:
        shuffle_types.append('type')
    if args.axon:
        shuffle_types.append('axon')
    if args.continuous:
        shuffle_types.append('continuous')

    if args.zscore and args.zscore not in shuffle_types:
        raise ValueError('zscore must be one of the shuffle types')
    
    main(args.trial_path, args.figures_path, shuffle_types, args.zscore, args.unique, args.dendrite, args.basal, 
         args.transparent, args.stats, args.verbose)
