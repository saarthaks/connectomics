import argparse
import pickle
import os

from core.Skeleton import Skeleton
from core.Branch import BranchSeq

def main(emst_path, output_path, include_unknown, verbose):
    with open(os.path.join(emst_path, 'all_emsts.pkl'), 'rb') as f:
        all_emsts = pickle.load(f)
    
    if verbose:
        print('Done loading emsts.')
    
    valid_types={'23P', '4P', '5P-IT', '5P-ET', '5P-NP', '6P-IT', '6P-CT'}
    if include_unknown:
        valid_types.add('Unknown')

    all_branches = {}
    for emst in all_emsts:
        paths = Skeleton.extract_paths_from_graph(emst, duplicate_tail=False)
        cell_id = emst.graph['cell_id']
        all_branches[cell_id] = [BranchSeq(path, emst, str(cell_id) + f'_{i}', valid_types=valid_types) for i, path in enumerate(paths)]
    
    if verbose:
        print('Done extracting branches.')
    
    with open(os.path.join(output_path, 'all_branches.pkl'), 'wb') as f:
        pickle.dump(all_branches, f)
    
    if verbose:
        print('Done saving branches.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--emst_path', type=str, help='Path to folder containing emst pickles')
    parser.add_argument('--output_path', type=str, help='Path to output folder')
    parser.add_argument('--include_unknown', action='store_true', help='Include unknown synapses')
    parser.add_argument('--verbose', action='store_true', help='Print progress statements')
    args = parser.parse_args()

    main(args.emst_path, args.output_path, args.include_unknown, args.verbose)