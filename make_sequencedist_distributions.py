import argparse
import pickle
import numpy as np
import os
from collections import defaultdict
from tqdm import tqdm

from core.plotting import *

def main(trial_path, figures_path, shuffle_types, xlim, pdf, cdf, basal=False, transparent=False, stats=False, verbose=False):
    
    with open(os.path.join(trial_path, 'original', 'trial_0_dists.pkl'), 'rb') as f:
        all_dists = pickle.load(f)
    
    real_dists = get_dists(all_dists, as_list=True, basal=basal)
    if verbose:
        print('Done counts for original distances.')
    
    all_shuffle_dists = defaultdict(list)
    for shuffle_type in shuffle_types:
        for fname in tqdm(os.listdir(os.path.join(trial_path, shuffle_type))):
            if fname.endswith('dists.pkl'):
                with open(os.path.join(trial_path, shuffle_type, fname), 'rb') as f:
                    dists = pickle.load(f)
                all_shuffle_dists[shuffle_type].append(get_dists(dists, as_list=True, basal=basal))
        
        if verbose:
            print(f'Done loading counts for shuffle type: {shuffle_type}.')
    
    
    if xlim:
        bins = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 30)
    else:
        bins = np.logspace(-1, 3, 30)

    str_comb = ''.join(shuffle_types)
    if basal:
        str_comb += '_basal'
    if pdf and cdf: 
        fig, _ = sequencedist_distribution(real_dists, all_shuffle_dists, bins=bins, histogram_types=shuffle_types, xlim=xlim, stats=stats)
        fig.savefig(os.path.join(figures_path, f'sequencedists_distribution_{str_comb}.png'), bbox_inches='tight', transparent=transparent)
        fig.savefig(os.path.join(figures_path, f'sequencedists_distribution_{str_comb}.svg'), bbox_inches='tight', transparent=transparent)
    elif pdf:
        fig, _ = sequencedist_pdf(real_dists, all_shuffle_dists, bins=bins, xlim=xlim, stats=stats)
        fig.savefig(os.path.join(figures_path, f'sequencedists_pdf_{str_comb}.png'), bbox_inches='tight', transparent=transparent)
        fig.savefig(os.path.join(figures_path, f'sequencedists_pdf_{str_comb}.svg'), bbox_inches='tight', transparent=transparent)
    elif cdf:
        fig, _ = sequencedist_cdf(real_dists, all_shuffle_dists, xlim=xlim, stats=stats)
        fig.savefig(os.path.join(figures_path, f'sequencedists_cdf_{str_comb}.png'), bbox_inches='tight', transparent=transparent)
        fig.savefig(os.path.join(figures_path, f'sequencedists_cdf_{str_comb}.svg'), bbox_inches='tight', transparent=transparent)
    else:
        raise ValueError('At least one of pdf or cdf must be True')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial_path', type=str, required=True)
    parser.add_argument('--figures_path', type=str, required=True)
    parser.add_argument('--basal', action='store_true')
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--type', action='store_true')
    parser.add_argument('--axon', action='store_true')
    parser.add_argument('--continuous', action='store_true')
    parser.add_argument('--pdf', action='store_true')
    parser.add_argument('--cdf', action='store_true')
    parser.add_argument('--xmin', default=2, type=float)
    parser.add_argument('--xmax', default=110, type=float)
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

    xlim = [args.xmin, args.xmax]
    main(args.trial_path, args.figures_path, shuffle_types, xlim, 
         args.pdf, args.cdf, basal=args.basal, transparent=args.transparent, 
         stats=args.stats, verbose=args.verbose)