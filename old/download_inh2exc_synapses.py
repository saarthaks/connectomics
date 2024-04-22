import argparse
import numpy as np
import pandas as pd
import pickle
import os
from time import sleep
from requests.exceptions import HTTPError

from tqdm import tqdm


from core.MicronsCAVE import CAVE

def download_input_synapses(client, post_pt_root_ids, pre_pt_root_ids=None, cell_df=None, rescale=True):
    if type(post_pt_root_ids) == int:
        post_pt_root_ids = [post_pt_root_ids]

    filter_dict = {'post_pt_root_id': post_pt_root_ids}
    if pre_pt_root_ids is not None:
        filter_dict['pre_pt_root_id'] = pre_pt_root_ids
    syn_df = client.download_synapses(filter_dict, cell_df=cell_df, rescale=rescale)

    if len(syn_df) >= 500000:
        chunk_1 = post_pt_root_ids[:len(post_pt_root_ids)//2]
        filter_dict_1 = {'post_pt_root_id': chunk_1}
        chunk_2 = post_pt_root_ids[len(post_pt_root_ids)//2:]
        filter_dict_2 = {'post_pt_root_id': chunk_2}
        syn_df_1 = client.download_synapses(filter_dict_1, cell_df=cell_df, rescale=rescale)
        syn_df_2 = client.download_synapses(filter_dict_2, cell_df=cell_df, rescale=rescale)
        syn_df = pd.concat([syn_df_1, syn_df_2], axis=0)

    return syn_df


def main(exc_cell_path, inh_cell_path, syn_table_path, verbose):

    client = CAVE()

    exc_cells = pd.read_csv(exc_cell_path)
    inh_cells = pd.read_csv(inh_cell_path)
    inh_cell_ids = inh_cells['pt_root_id'].values
    
    if verbose:
        print('Downloading synapses...')
    
    timeout = 600
    chunk_size = 600

    num_chunks = int(np.ceil((len(exc_cells['pt_root_id'].values))/chunk_size))
    all_syn_dfs = []
    for chunk in tqdm(range(num_chunks)):
        chunk_ids = exc_cells['pt_root_id'].values[chunk*chunk_size:(chunk+1)*chunk_size]
        try:
            syn_df = download_input_synapses(client, chunk_ids, pre_pt_root_ids=inh_cell_ids, cell_df=exc_cells, rescale=True)
        except HTTPError:
            print(f"Chunk {chunk} failed, retrying")
            sleep(timeout)
            syn_df = download_input_synapses(client, chunk_ids, pre_pt_root_ids=inh_cell_ids, cell_df=exc_cells, rescale=True)
        print(len(syn_df))
        all_syn_dfs.append(syn_df)

    syn_df = pd.concat(all_syn_dfs, axis=0)
    syn_df.to_csv(syn_table_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download synapses from inhibitory cells onto excitatory cells')
    parser.add_argument('--exc_cell_path', type=str, help='Path to excitatory cell list')
    parser.add_argument('--inh_cell_path', type=str, help='Path to inhibitory cell list')
    parser.add_argument('--syn_table_path', type=str, help='Path to save synapse table')
    parser.add_argument('--verbose', action='store_true', help='Print progress')
    args = parser.parse_args()

    main(args.exc_cell_path, args.inh_cell_path, args.syn_table_path, args.verbose)