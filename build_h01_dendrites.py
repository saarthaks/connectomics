import argparse
import pandas as pd
import pickle
import os
import navis
import cloudvolume as cv
import networkx as nx
import numpy as np
from tqdm import tqdm

from core.Tree import Tree
from core.Branch import BranchSeq


def filter_and_connect_graph(original_graph, desired_nodes):
    # Step 1: Initially, identify nodes with multiple parents and nodes to keep
    nodes_with_multiple_children = {node for node in original_graph.nodes() if original_graph.out_degree(node) > 1}
    nodes_to_keep = desired_nodes.union(nodes_with_multiple_children).union({-1})  # Include root node

    # Create a copy of the graph to work on
    G = original_graph.copy()

    # Step 2: For nodes not in the keep list, redirect parents to children and remove the node
    for node in list(G.nodes()):  # List conversion to avoid modification during iteration
        if node not in nodes_to_keep:
            parents = list(G.predecessors(node))
            children = list(G.successors(node))
            for parent in parents:
                for child in children:
                    G.add_edge(parent, child)  # Connect parent directly to child
            G.remove_node(node)  # Remove the node after re-connecting

    assert -1 in G.nodes(), "Root node not found in the graph"
    return G

def main(data_path):
    navis.patch_cloudvolume()
    vol = cv.CloudVolume('precomputed://gs://h01-release/data/20210601/c3', use_https=True, progress=True, parallel=True)

    syn_df = pd.read_csv('/home/saarthak/microns/data/syn_df.csv')    
    cell_df = pd.read_csv('/home/saarthak/microns/data/post_ids.csv')

    post_id_counts = syn_df['post_pt_root_id'].value_counts()

    # sort post_id_df by post_id_counts
    cell_df['count'] = cell_df['pt_root_id'].apply(lambda x: post_id_counts[x])
    cell_df = cell_df.sort_values(by='count', ascending=False)
    
    chunk_size = 10

    cell_ids = cell_df['pt_root_id'].values
    N = len(cell_ids)

    starting_chunk = 270
    cell_ids = cell_ids[starting_chunk*chunk_size:]

    # all_emsts = {}
    all_trees = {}
    all_branches = {}
    # chunk cell_ids int cids by chunk_size and loop
    for i in tqdm(range(starting_chunk, int(np.ceil(N/chunk_size)))):
        cid = cell_ids[i*chunk_size:(i+1)*chunk_size]
        nrns = vol.mesh.get(cid, as_navis=True)
        nrns = navis.simplify_mesh(nrns, F=1/3, parallel=True)
        sks = navis.skeletonize(nrns, parallel=True)
        sks = navis.heal_skeleton(sks, parallel=True)
        sks = navis.prune_twigs(sks, 6000, parallel=True)
        trees = {}
        branches = {}
        for skp in sks:
            syn_pos = np.array(syn_df[syn_df['post_pt_root_id'] == skp.id][['x', 'y', 'z']].values) * np.array([8, 8, 33])
            pre_cell_ids = np.array(syn_df[syn_df['post_pt_root_id'] == skp.id]['pre_pt_root_id'].values)
            syn_ids = np.array(syn_df[syn_df['post_pt_root_id'] == skp.id].index)
            
            segment_length_dict = {node: navis.segment_length(skp, seg)/1000 for seg in skp.segments for node in seg}
            node_ids,_ = skp.snap(syn_pos)
            RG = skp.get_graph_nx().reverse()

            root_pos = np.array(skp.nodes.iloc[skp.root][['x', 'y', 'z']].values[0])

            RG.add_node(-1, pos=root_pos/1000)
            for r in skp.root:
                RG.add_edge(r, -1)
            G = filter_and_connect_graph(RG, set(node_ids))
            G.graph['cell_id'] = skp.id
            branch_lengths = [segment_length_dict.get(node, 0) for node in list(G.nodes)]
            # set node attributes of G with syn_pos, cell_types, and pre_cell_ids
            nx.set_node_attributes(G, dict(zip(node_ids, syn_pos/1000)), 'pos')
            nx.set_node_attributes(G, dict(zip(node_ids, len(node_ids)*['4P'])), 'cell_type')
            nx.set_node_attributes(G, dict(zip(node_ids, pre_cell_ids)), 'pre_cell_id')
            nx.set_node_attributes(G, dict(zip(list(G.nodes), branch_lengths)), 'branch_length')
            
            tree = Tree(G.reverse(), root_id=-1)
            trees[skp.id] = tree
            branches[skp.id] = [BranchSeq(path, tree.graph, (cid, j)) for j, path in enumerate(tree.get_paths())]


        with open(os.path.join(data_path, 'trees', f'trees_{i}.pkl'), 'wb') as f:
            pickle.dump(trees, f)
        with open(os.path.join(data_path, 'branches', f'branches_{i}.pkl'), 'wb') as f:
            pickle.dump(branches, f)

        all_trees.update(trees)
        all_branches.update(branches)
    
    # with open(os.path.join(emst_path, 'all_emsts.pkl'), 'wb') as f:
    #     pickle.dump(all_emsts, f)

    with open(os.path.join(data_path, 'trees', 'all_trees.pkl'), 'wb') as f:
        pickle.dump(all_trees, f)
    
    with open(os.path.join(data_path, 'branches', 'all_branches.pkl'), 'wb') as f:
        pickle.dump(all_branches, f)


if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # # parser.add_argument('--data_path', type=str, required=True)
    # args = parser.parse_args()

    # main(args.data_path)
    main('/home/saarthak/microns/data')
