import argparse
import pandas as pd
import pickle
import os
import navis
import cloudvolume as cv
import numpy as np
import networkx as nx
from tqdm import tqdm
from sklearn.mixture import GaussianMixture

from core.Skeleton import Skeleton

def fit_gmm(paths, skp, min_path_length=4, single_syn_std=1):
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

def process_chunk(layer_vol_list, i, cid, vol, syn_df, data_path):

    # get all meshes in parallel, then skeletonize in parallel
    nrns = vol.mesh.get(cid, as_navis=True)
    nrns = navis.simplify_mesh(nrns, F=1/3, parallel=True)
    sks = navis.skeletonize(nrns, parallel=True)
    sks = navis.prune_twigs(sks, 6000, parallel=True)

    axons = {}
    gmms = {}
    for skp in sks:

        # get synapse positions on axon and snap to nodes of skeleton
        syn_pos = np.array(syn_df[syn_df['pre_pt_root_id'] == skp.id][['x', 'y', 'z']].values) * np.array([8, 8, 33])
        syn_ids = syn_df[syn_df['pre_pt_root_id'] == skp.id]['syn_id'].values
        node_ids,_ = skp.snap(syn_pos)
        RG = skp.get_graph_nx().reverse()

        root_pos = np.array(skp.nodes.iloc[skp.root][['x', 'y', 'z']].values[0] / 1000)
        all_node_positions = np.array(skp.nodes[['x', 'y', 'z']].values) / 1000
        all_node_ids = skp.nodes['node_id'].values

        for i, n_layer in enumerate(layer_vol_list):
            if n_layer.contains(all_node_positions):
                layer = i + 1  # Adjusted for 1-based indexing
                break

        RG.add_node(-1, pos=root_pos)
        for r in skp.root:
            RG.add_edge(r, -1)
        G = Skeleton.filter_and_connect_graph(RG, set(node_ids))
        nx.set_node_attributes(G, dict(zip(node_ids, syn_ids)), 'syn_ids')
        nx.set_node_attributes(G, dict(zip(all_node_ids, all_node_positions)), 'pos')
        nx.set_node_attributes(G, dict(zip(all_node_ids, layer)), 'layer')


        # get all segments that contain at least one outgoing synapse
        axon_paths = [path for path in skp.small_segments if len(set(node_ids).intersection(path)) > 0]

        # git a gmm to the list of paths
        axons[skp.id] = G
        gmms[skp.id] = fit_gmm(axon_paths, skp)

    return gmms, axons

def main(data_path):
    navis.patch_cloudvolume()
    vol = cv.CloudVolume('precomputed://gs://h01-release/data/20210601/c3', use_https=True, progress=True, parallel=True)

    layer_vol = cv.CloudVolume('precomputed://gs://h01-release/data/20210601/layers', use_https=True, progress=False)
    layer_vol_list = [layer_vol.mesh.get(i, as_navis=True) for i in range(1, 8)]
    layer_vol_list = [navis.Volume.from_object(layer.trimesh[0]) for layer in layer_vol_list]

    syn_df = pd.read_csv('./data/syn_df.csv')    
    cell_df = pd.read_csv('./data/pre_ids.csv')

    chunk_size = 10

    cell_ids = list(cell_df['pt_root_id'].values)

    starting_chunk = 0
    all_gmms = {}
    all_axons = {}
    # chunk cell_ids int cids by chunk_size and loop
    for i in tqdm(range(starting_chunk, int(np.ceil(len(cell_ids)/chunk_size)))):
        cid = cell_ids[i*chunk_size:(i+1)*chunk_size]
        gmms, axons = process_chunk(layer_vol_list, i, cid, vol, syn_df, data_path)

        with open(os.path.join(data_path, 'gmms', f'gmms_{i}.pkl'), 'wb') as f:
            pickle.dump(gmms, f)
        with open(os.path.join(data_path, 'axons', f'axons_{i}.pkl'), 'wb') as f:
            pickle.dump(axons, f)

        all_gmms.update(gmms)
        all_axons.update(axons)

    with open(os.path.join(data_path, 'gmms', 'all_gmms.pkl'), 'wb') as f:
        pickle.dump(all_gmms, f)
    
    with open(os.path.join(data_path, 'axons', 'all_axons.pkl'), 'wb') as f:
        pickle.dump(all_axons, f)
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    args = parser.parse_args()

    main(args.data_path)
