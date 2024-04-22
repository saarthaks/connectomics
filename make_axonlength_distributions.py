import argparse
import pickle
import numpy as np
import os
from collections import defaultdict
from tqdm import tqdm

from core.plotting import *

def main(trial_path, figures_path, shuffle_types, pdf, cdf, xlim, transparent, stats, verbose):
    
    with open(os.path.join(trial_path, 'original', 'trial_0_axonlengths.pkl'), 'rb') as f:
        real_axon_lengths = pickle.load(f)
    
    all_shuffle_lengths = defaultdict(list)
    for shuffle_type in shuffle_types:
        for fname in tqdm(os.listdir(os.path.join(trial_path, shuffle_type))):
            if fname.endswith('_axonlengths.pkl'):
                with open(os.path.join(trial_path, shuffle_type, fname), 'rb') as f:
                    lengths = pickle.load(f)
                all_shuffle_lengths[shuffle_type].append(lengths)
    
    if verbose:
        print('Finished loading axon lengths.')
    
    bins = np.logspace(np.log10(0.08), np.log10(30), 30)
    if pdf and cdf: 
        fig, _ = axonlength_distribution(real_axon_lengths, all_shuffle_lengths, bins=bins, histogram_types=shuffle_types, xlim=xlim, stats=stats)
        fname = ''.join(shuffle_types)
        fig.savefig(os.path.join(figures_path, f'axonlengths_distribution_{fname}.png'), bbox_inches='tight', transparent=transparent)
        fig.savefig(os.path.join(figures_path, f'axonlengths_distribution_{fname}.svg'), bbox_inches='tight', transparent=transparent)
    elif pdf:

        fig, _ = axonlength_pdf(real_axon_lengths, all_shuffle_lengths, bins=bins, xlim=xlim, stats=stats)
        fig.savefig(os.path.join(figures_path, 'axonlengths_pdf.png'), bbox_inches='tight', transparent=transparent)
        fig.savefig(os.path.join(figures_path, 'axonlengths_pdf.svg'), bbox_inches='tight', transparent=transparent)
    elif cdf:
        fig, _ = axonlength_cdf(real_axon_lengths, all_shuffle_lengths, xlim=xlim, stats=stats)
        fig.savefig(os.path.join(figures_path, 'axonlengths_cdf.png'), bbox_inches='tight', transparent=transparent)
        fig.savefig(os.path.join(figures_path, 'axonlengths_cdf.svg'), bbox_inches='tight', transparent=transparent)
    else:
        raise ValueError('At least one of pdf or cdf must be True')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial_path', type=str, required=True)
    parser.add_argument('--figures_path', type=str, required=True)
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--type', action='store_true')
    parser.add_argument('--axon', action='store_true')
    parser.add_argument('--continuous', action='store_true')
    parser.add_argument('--pdf', action='store_true')
    parser.add_argument('--cdf', action='store_true')
    parser.add_argument('--xmin', default=0.08, type=float)
    parser.add_argument('--xmax', default=30, type=float)
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
    elif args.continuous:
        shuffle_types.append('continuous')
    
    xlim = [args.xmin, args.xmax]
    main(args.trial_path, args.figures_path, shuffle_types, 
         args.pdf, args.cdf, xlim, args.transparent, args.stats, args.verbose)