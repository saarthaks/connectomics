import argparse
import pandas as pd
import pickle
import os
import navis
import navis.interfaces.microns as mi
import networkx as nx
import numpy as np
from tqdm import tqdm
from collections import defaultdict


from core.MicronsCAVE import CAVE
from core.Skeleton import Skeleton
from core.Tree import Tree
from core.Branch import BranchSeq


def filter_and_connect_graph(original_graph, desired_nodes):
    # Step 1: Initially, identify nodes with multiple parents and nodes to keep
    nodes_with_multiple_parents = {node for node in original_graph.nodes() if original_graph.in_degree(node) > 1}
    nodes_to_keep = desired_nodes.union(nodes_with_multiple_parents).union({-1})  # Include root node

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

def main(emst_path, verbose):
    og_msts_dict = defaultdict(list)
    with open('/drive_sdc/ssarup/connectomics_data/data/all_msts.pkl', 'rb') as f:
        og_msts = pickle.load(f)
        for skel in og_msts:
            og_msts_dict[skel.graph['cell_id']].append(skel)
    
    cell_df = pd.read_csv('/drive_sdc/ssarup/microns_data/exc_cells.csv')

    if verbose:
        print('Downloading synapses...')
    
    timeout = 1200
    chunk_size = 100

    cell_ids = list(og_msts_dict.keys())
    N = len(cell_ids)
    # with open('./soma_missing.pkl', 'rb') as f:
    #     cell_ids = pickle.load(f)

    starting_chunk = 0
    cell_ids = cell_ids[starting_chunk*chunk_size:]

    # all_emsts = {}
    all_trees = {}
    all_branches = {}
    # chunk cell_ids int cids by chunk_size and loop
    for i in tqdm(range(starting_chunk, int(np.ceil(N/chunk_size)))):
        cid = cell_ids[i*chunk_size:(i+1)*chunk_size]
        nrns = mi.fetch_neurons(cid, lod=3, with_synapses=False)
        sks = navis.skeletonize(nrns)
        # emsts = {}
        trees = {}
        branches = {}
        for skel in sks:
            pre_cell_ids = list(nx.get_node_attributes(og_msts_dict[skel.id][0], 'pre_cell_id').values())
            cell_types = list(nx.get_node_attributes(og_msts_dict[skel.id][0], 'cell_type').values())
            syn_pos = np.array(list(nx.get_node_attributes(og_msts_dict[skel.id][0], 'pos').values())) * np.array([1000, 1000, 1000])
            
            skp = skel.prune_twigs(6000)
            segment_length_dict = {node: navis.segment_length(skp, seg)/1000 for seg in skp.segments for node in seg}
            node_ids,_ = skp.snap(syn_pos)
            RG = skp.get_graph_nx()

            # get cell_df row whose pt_root_id == skel.id
            cell_df_row = cell_df[cell_df.pt_root_id == skel.id]
            # get position from columns pt_x, pt_y, pt_z, as np.array
            root_pos = np.array(cell_df_row[['pt_x', 'pt_y', 'pt_z']].values[0])

            # root_pos = skp.nodes.query('node_id == @skp.soma')[['x', 'y', 'z']].values.mean(axis=0)
            RG.add_node(-1, pos=root_pos)
            for r in skp.root:
                RG.add_edge(r, -1)
            G = filter_and_connect_graph(RG, set(node_ids))
            G.graph['cell_id'] = skel.id
            branch_lengths = [segment_length_dict.get(node, 0) for node in list(G.nodes)]
            # set node attributes of G with syn_pos, cell_types, and pre_cell_ids
            nx.set_node_attributes(G, dict(zip(node_ids, syn_pos/1000)), 'pos')
            nx.set_node_attributes(G, dict(zip(node_ids, cell_types)), 'cell_type')
            nx.set_node_attributes(G, dict(zip(node_ids, pre_cell_ids)), 'pre_cell_id')
            nx.set_node_attributes(G, dict(zip(list(G.nodes), branch_lengths)), 'branch_length')
            
            tree = Tree(G, root_id=-1)
            trees[skel.id] = tree
            branches[skel.id] = [BranchSeq(path, tree.graph, (cid, j)) for j, path in enumerate(tree.get_paths())]

        # if verbose:
        #     print('Done skeletonizing cells.')

        with open(os.path.join(emst_path, f'trees_{i}.pkl'), 'wb') as f:
            pickle.dump(trees, f)
        with open(os.path.join(emst_path, f'branches_{i}.pkl'), 'wb') as f:
            pickle.dump(branches, f)

        # if verbose:
        #     print('Done saving msts.')

        # all_emsts.update(emsts)
        all_trees.update(trees)
        all_branches.update(branches)
    
    # with open(os.path.join(emst_path, 'all_emsts.pkl'), 'wb') as f:
    #     pickle.dump(all_emsts, f)

    with open(os.path.join(emst_path, 'all_trees.pkl'), 'wb') as f:
        pickle.dump(all_trees, f)
    
    with open(os.path.join(emst_path, 'all_branches.pkl'), 'wb') as f:
        pickle.dump(all_branches, f)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--emst_path', type=str, required=True)
    parser.add_argument('--verbose', type=bool, default=False)
    args = parser.parse_args()

    main(args.emst_path, args.verbose)
