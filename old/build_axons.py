import argparse
import pandas as pd
import pickle
import os

from core.MicronsCAVE import CAVE
from core.Skeleton import Skeleton
from core.AxonModel import AxonModel

def main(cell_path, axon_path, gmm_path, syn_k, soma_k, twig_length, single_syn_std, verbose):
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
    chunk_size = 750

    i = 0
    for synapses_grouped in client.download_output_synapses_list(list(exc_cells.pt_root_id.values), 
                                                                 cell_df=exc_cells, 
                                                                 timeout=timeout, 
                                                                 chunk_size=chunk_size):
        axns = []
        gmms = []
        for cell_id, syn_group in synapses_grouped:
            axn = AxonModel(exc_cells[exc_cells['pt_root_id']==cell_id], syn_group, 
                            syn_k=syn_k, soma_k=soma_k, 
                            twig_length=twig_length, single_syn_std=single_syn_std)
            axns.append(axn.smooth_mst)
            gmms.append(axn.gmm)
        
        if verbose:
            print('Done skeletonizing axons.')
        
        with open(os.path.join(axon_path, f'axon_{i}.pkl'), 'wb') as f:
            pickle.dump(axns, f)
        with open(os.path.join(gmm_path, f'gmm_{i}.pkl'), 'wb') as f:
            pickle.dump(gmms, f)

        if verbose:
            print('Done saving axons.')
        
        i+=1
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell_path', type=str, required=True)
    parser.add_argument('--axon_path', type=str, required=True)
    parser.add_argument('--gmm_path', type=str, required=True)
    parser.add_argument('--syn_k', type=int, default=8)
    parser.add_argument('--soma_k', type=int, default=8)
    parser.add_argument('--twig_length', type=int, default=4)
    parser.add_argument('--single_syn_std', type=int, default=5)
    parser.add_argument('--verbose', default=False)

    args = parser.parse_args()
    main(args.cell_path, args.axon_path, args.gmm_path, 
         args.syn_k, args.soma_k, args.twig_length, args.single_syn_std, 
         args.verbose)
