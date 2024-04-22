import argparse
import pickle
import os
from tqdm import tqdm

from core.Skeleton import Skeleton
from core.Branch import BranchSeq
from core.Tree import Tree

def main(trees_path, output_path, verbose):
    with open(trees_path, 'rb') as f:
        all_trees_dict = pickle.load(f)
    
    if verbose:
        print('Done loading trees.')

    all_branches = {}
    for cell_id, tree in tqdm(all_trees_dict.items()):
        paths = tree.get_paths()
        all_branches[cell_id] = [BranchSeq(path, tree.graph, hash((cell_id, j))) for j, path in enumerate(paths)]
    
    if verbose:
        print('Done extracting branches.')
    
    with open(output_path, 'wb') as f:
        pickle.dump(all_branches, f)
    
    if verbose:
        print('Done saving branches.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trees_path', type=str, help='Path to folder containing trees pickles')
    parser.add_argument('--output_path', type=str, help='Path to output folder')
    parser.add_argument('--verbose', action='store_true', help='Print progress statements')

    args = parser.parse_args()

    main(args.trees_path, args.output_path, args.verbose)
