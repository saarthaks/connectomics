import argparse
import pandas as pd
import pickle
import os

from core.MicronsCAVE import CAVE
from core.Skeleton import Skeleton


def main(cell_path, mst_path, emst_path, syn_k, soma_k, twig_length, verbose):
    client = CAVE()

    try:
        exc_cells = pd.read_csv(cell_path)
    except FileNotFoundError:
        print('Cell list not found. Downloading...')
        exc_cells = client.download_excitatory_cells()
        exc_cells.to_csv(cell_path, index=False)
        if verbose:
            print('Done downloading cells.')
    
    if verbose:
        print('Downloading synapses...')
    
    timeout = 600
    chunk_size = 150

    i = 0
    for synapses_grouped in client.download_input_synapses_list(list(exc_cells.pt_root_id.values), 
                                                                 cell_df=exc_cells, 
                                                                 timeout=timeout, 
                                                                 chunk_size=chunk_size):
        msts = []
        emsts = []
        for cell_id, syn_group in synapses_grouped:
            skel = Skeleton(exc_cells[exc_cells['pt_root_id']==cell_id], syn_group, syn_k=syn_k, soma_k=soma_k)
            msts.append(skel.mst)
            skel.smooth(twig_length=twig_length, prune_unknown=True)
            emsts.append(skel.extract_excitatory_smooth_mst())

        if verbose:
            print('Done skeletonizing cells.')

        with open(os.path.join(mst_path, f'mst_{i}.pkl'), 'wb') as f:
            pickle.dump(msts, f)
        with open(os.path.join(emst_path, f'emst_{i}.pkl'), 'wb') as f:
            pickle.dump(emsts, f)

        if verbose:
            print('Done saving msts.')

        i+=1
    
    all_emsts = []
    # loop through all files in emst_path, load them, and append to all_emsts
    for file in os.listdir(emst_path):
        with open(os.path.join(emst_path, file), 'rb') as f:
            emsts = pickle.load(f)
            all_emsts.extend(emsts)

    with open(os.path.join(emst_path, 'all_emsts.pkl'), 'wb') as f:
        pickle.dump(all_emsts, f)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cell_path', type=str, required=True)
    parser.add_argument('--mst_path', type=str, required=True)
    parser.add_argument('--emst_path', type=str, required=True)
    parser.add_argument('--syn_k', type=int, default=6)
    parser.add_argument('--soma_k', type=int, default=12)
    parser.add_argument('--twig_length', type=int, default=2)
    parser.add_argument('--verbose', type=bool, default=False)
    args = parser.parse_args()

    main(args.cell_path, args.mst_path, args.emst_path, 
         args.syn_k, args.soma_k, args.twig_length, 
         args.verbose)