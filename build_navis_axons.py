import argparse
import pandas as pd
import pickle
import os
import navis
import navis.interfaces.microns as mi
import networkx as nx
import numpy as np
from time import sleep
from tqdm import tqdm
from collections import defaultdict
from sklearn.mixture import GaussianMixture


def fit_gmm(paths, skp, min_path_length=4, single_syn_std=0.5):
    all_positions = []
    all_means = []
    all_precisions = []
    for path in paths:
        path_positions = np.array(skp.nodes[skp.nodes['node_id'].isin(path)][['x', 'y', 'z']].values) / 1000
        mean = np.mean(path_positions, axis=0)
        if len(path_positions) >= min_path_length:
            precision = np.linalg.inv(np.cov(path_positions.T))
        elif len(path_positions) == 1:
            precision = np.diag(1/(np.array(3*[single_syn_std]))**2)
        else:
            variance = np.var(path_positions, axis=0)
            variance[variance == 0] = single_syn_std**2
            precision = np.diag(1/variance)
        
        precision[np.isinf(precision)] = 1/(single_syn_std**2)
        all_positions.append(path_positions)
        all_means.append(mean)
        all_precisions.append(precision)

    all_positions = np.concatenate(all_positions, axis=0)
    gmm = GaussianMixture(n_components=len(all_means), covariance_type='full', 
            means_init=np.array(all_means),
            precisions_init=np.array(all_precisions))
    if all_positions.shape[0] == 1:
        all_positions = np.concatenate([all_positions, all_positions], axis=0)
    gmm.fit(all_positions)
    return gmm

def process_chunk(i, cid, client, data_path):
    nrns = mi.fetch_neurons(cid, lod=3, with_synapses=False)
    sks = navis.skeletonize(nrns)
    gmms = {}
    for skel in sks:
        pre = client.materialize.synapse_query(pre_ids=skel.id)
        pre['type'] = 'pre'
        locs = np.vstack(pre['pre_pt_position'].values)
        locs = locs * np.array([4, 4, 40])
        skp = skel.prune_twigs(6000)
        node_ids,_ = skp.snap(locs)
        axon_paths = [path for path in skp.small_segments if len(set(node_ids).intersection(path)) > 0]
        gmms[skel.id] = fit_gmm(axon_paths, skp)
    
    with open(os.path.join(data_path, f'gmms_{i}.pkl'), 'wb') as f:
        pickle.dump(gmms, f)

    return gmms

def main(data_path):
    client = mi.get_cave_client()

    timeout = 1200
    chunk_size = 100

    cell_df = pd.read_csv('/drive_sdc/ssarup/microns_data/navis_exc_cells.csv')
    cell_ids = list(cell_df['pt_root_id'].values)

    starting_chunk = 0
    cell_ids = cell_ids[starting_chunk*chunk_size:]

    all_gmms = {}
    # chunk cell_ids int cids by chunk_size and loop
    for i in tqdm(range(starting_chunk, int(np.ceil(len(cell_ids)/chunk_size)))):
        cid = cell_ids[i*chunk_size:(i+1)*chunk_size]
        try:
            gmms = process_chunk(i, cid, client, data_path)
        except Exception as e:
            print(f'Error in chunk {i}: {e}')
            sleep(timeout)
            gmms = process_chunk(i, cid, client, data_path)
        all_gmms.update(gmms)

    with open(os.path.join(data_path, 'all_gmms.pkl'), 'wb') as f:
        pickle.dump(all_gmms, f)
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    args = parser.parse_args()

    main(args.data_path)
