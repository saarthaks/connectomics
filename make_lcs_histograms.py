import argparse
import pickle
import os
from collections import defaultdict
from tqdm import tqdm

import matplotlib.pyplot as plt

shuffle_type_map = {
    'random': 'Random Shuffle',
    'type': 'Cell-type Shuffle',
    'axon': 'Axon-biased Shuffle'
}

def get_counts(dists):
    counts = defaultdict(int)
    for cell_id in dists:
        for branch_id in dists[cell_id]:
            lcs_dists = [pair[0] for pair in dists[cell_id][branch_id]]
            max_lcs = max(lcs_dists)
            if max_lcs > 1:
                counts[max_lcs] += 1

    return counts

def main(trial_path, figures_path, random, type, axon, verbose=False):
    
    shuffle_types = []
    if random:
        shuffle_types.append('random')
    if type:
        shuffle_types.append('type')
    if axon:
        shuffle_types.append('axon')
    
    with open(os.path.join(trial_path, 'original', 'trial_0_dists.pkl'), 'rb') as f:
        all_dists = pickle.load(f)
    

    real_counts = get_counts(all_dists)
    if verbose:
        print('Done counts for original distances.')
    
    all_shuffle_counts = {}
    for shuffle_type in shuffle_types:
        # iterate over all trials in shuffle_type folder, 
        # load distances, and 
        # update the counts dict for that shuffle type

        shuffle_counts = defaultdict(list)
        for fname in tqdm(os.listdir(os.path.join(trial_path, shuffle_type))):
            if fname.endswith('dists.pkl'):
                with open(os.path.join(trial_path, shuffle_type, fname), 'rb') as f:
                    dists = pickle.load(f)
                counts = get_counts(dists)
                for k, v in counts.items():
                    shuffle_counts[k].append(v)
        
        all_shuffle_counts[shuffle_type] = shuffle_counts
        if verbose:
            print(f'Done loading counts for shuffle type: {shuffle_type}.')
    
    bins = 5
    for key in real_counts:
        plt.figure(figsize=(6,3))
        plt.axvline(real_counts[key], color='k', linestyle='dashed', linewidth=2, label='Real Data')
        for shuffle_type in shuffle_types:
            if key in all_shuffle_counts[shuffle_type]:
                plt.hist(all_shuffle_counts[shuffle_type][key], bins=bins, alpha=0.5, edgecolor='black', label=shuffle_type_map[shuffle_type])

        plt.xlabel('Number of Repeated Subsequences', fontsize=16)
        plt.xticks(fontsize=14)
        plt.tick_params(axis='x', which='major', length=5)
        plt.tick_params(axis='y', which='both', left=False, labelleft=False)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        str_comb = ''.join(shuffle_types)
        plt.savefig(os.path.join(figures_path, f'lcs_histogram_{key}_{str_comb}.svg'), bbox_inches='tight', transparent=True)
        plt.savefig(os.path.join(figures_path, f'lcs_histogram_{key}_{str_comb}.png'), bbox_inches='tight', transparent=True)
        plt.close()
    
        if verbose:
            print(f'Done saving histogram for LCS={key}.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial_path', type=str, required=True)
    parser.add_argument('--figures_path', type=str, required=True)
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--type', action='store_true')
    parser.add_argument('--axon', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    main(args.trial_path, args.figures_path, 
         args.random, args.type, args.axon, 
         args.verbose)