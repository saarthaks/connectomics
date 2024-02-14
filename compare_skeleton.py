from core.MicronsCAVE import CAVE
from scipy.sparse.csgraph import breadth_first_order
from scipy.sparse import csr_matrix
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from zss import simple_distance
from nltk.metrics.distance import edit_distance
import argparse
import numpy as np


class Node:
    def __init__(self, label):
        self.label = label
        self.children = []

def build_zss_tree(nx_tree, root_label):
    root = Node(root_label)
    _build_tree_helper(nx_tree, root, root_label, set())
    return root

def _build_tree_helper(nx_tree, current_node, current_label, visited):
    visited.add(current_label)
    children = []
    for neighbor in nx_tree.neighbors(current_label):
        if neighbor not in visited:
            child_node = Node(neighbor)
            children.append(child_node)
            _build_tree_helper(nx_tree, child_node, neighbor, visited)
    current_node.children = sorted(children, key=lambda x: x.label) # sorts node!


def all_paths_to_leaves(tree, root):
    paths = []
    for leaf in [x for x in tree.nodes() if tree.degree(x) == 1 and x != root]:
        paths.append(nx.shortest_path(tree, root, leaf))
    return paths

def min_edit_distance(paths1, paths2):
    min_edit_dic= {}
    for path1 in paths1:
        for path2 in paths2:
            
            distance = edit_distance(path1, path2)/(len(path1)-1)
            if str(path1) not in min_edit_dic:
                min_edit_dic[str(path1)] = distance
            else:
                min_edit_dic[str(path1)]  = min(min_edit_dic[str(path1)], distance)
    return min_edit_dic

def reindex_tree_by_attribute(tree, attribute='syn_id'):
    '''
    This function reindex the tree with syn_id for easier comparision
    '''
    new_tree = type(tree)()
    for node, data in tree.nodes(data=True):
        if attribute in data:
            new_tree.add_node(data[attribute])
        else:
            raise ValueError(f"Node {node} does not have the attribute '{attribute}'.")
    for u, v in tree.edges():
        new_u = tree.nodes[u][attribute]
        new_v = tree.nodes[v][attribute]
        new_tree.add_edge(new_u, new_v)
    return new_tree

def draw_trees_side_by_side(tree1, tree2, title1, title2, path = None):
    '''
    This function draws the two trees side by side
    '''
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    pos = nx.spring_layout(tree1)  
    nx.draw(tree1, pos, with_labels=True, ax=axs[0], node_size=200, node_color='skyblue',  font_size = 8)
    axs[0].set_title(title1)

    pos = nx.spring_layout(tree2)
    nx.draw(tree2, pos, with_labels=True, ax=axs[1], node_size=200, node_color='lightgreen', font_size = 8)
    axs[1].set_title(title2)

    if path:
        plt.savefig(path)
    plt.show()

def reconstruct_graph(df, csgraph, root_id):
    '''
    This function is used to prune the csgraph created by the skeleton
    '''
    valid_ids = set(df['skeleton_id'])
    num_nodes = csgraph.shape[0]
    node_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(valid_ids.union({root_id})))}
    rows = []
    cols = []
    data = []
    order, predecessors = breadth_first_order(csgraph, i_start=root_id, directed=False, return_predecessors=True)
    for old_id in range(num_nodes):
        if old_id in valid_ids or old_id == root_id:
            new_id = node_mapping[old_id]
            ancestor_id = predecessors[old_id]
            while ancestor_id != -9999 and ancestor_id not in valid_ids and ancestor_id != root_id:
                ancestor_id = predecessors[ancestor_id]
            if ancestor_id == -9999:
                ancestor_id = root_id
            new_ancestor_id = node_mapping[ancestor_id]
            if new_id != new_ancestor_id:
                rows.append(new_id)
                cols.append(new_ancestor_id)
                data.append(1) 
    new_graph = csr_matrix((data, (rows, cols)), shape=(len(node_mapping), len(node_mapping)), dtype=np.float32)

    new_tree = nx.from_scipy_sparse_array(new_graph)
    for old_index, new_index in node_mapping.items():
        syn_id = df.loc[df['skeleton_id'] == old_index, 'syn_id'].values[0]
        nx.set_node_attributes(new_tree, {new_index: syn_id}, 'syn_id')

    return new_tree

def prune_mst(mst):
    '''
    This function creates the reference skeleton and prunes mst accordingly
    After pruning, it does not contain synapses that has non-unique skeleton_id
    '''

    example_cell_id = int(mst.graph["cell_id"])
    print(example_cell_id)
    client = CAVE()
    sk_df, sk_csgraph, rood_id_csgraph = client.download_sk_anno(example_cell_id)

    syn_id_wanted = []
    non_unique_skeleton_ids = sk_df[sk_df.duplicated('skeleton_id', keep=False)]['skeleton_id'].unique()
    filtered_df_unique = sk_df[~sk_df['skeleton_id'].isin(non_unique_skeleton_ids) | (sk_df['syn_id'] == -1)]

    kept_syn_ids = list(filtered_df_unique["syn_id"])
    for node in mst.nodes(data=True):
        node_id, attrs = node
        if 'syn_id' in attrs and attrs['syn_id'] in kept_syn_ids:
            syn_id_wanted.append(attrs['syn_id'])

    nodes_to_remove = []
    for node in mst.nodes():
        syn_id = mst.nodes[node]['syn_id']
        if syn_id not in syn_id_wanted:
            nodes_to_remove.append(node)

    # assuming the mst is directed 
    for node in nodes_to_remove:
        ancestors = list(nx.ancestors(mst, node))
        children = list(mst.successors(node))
        for child in children:
            for ancestor in ancestors:
                if ancestor not in nodes_to_remove:
                    mst.add_edge(ancestor, child)
                    break

    mst.remove_nodes_from(nodes_to_remove)
    mst = mst.to_undirected()
    print("Is the MST still a valid tree: ", nx.is_tree(mst))
    print("Number of Nodes: ", len(mst.nodes))

    return mst, syn_id_wanted, sk_df, sk_csgraph, rood_id_csgraph


def prune_ref_tree(syn_id_wanted, sk_df, sk_csgraph, rood_id_csgraph, save_raw = False, file_name = None):

    '''
    This funciton prunes the reference tree
    with an option to save the raw tree (tree with all synapses, not just excitatory)
    '''

    filtered_df = sk_df[sk_df['syn_id'].isin(syn_id_wanted)]
    new_tree = reconstruct_graph(filtered_df, sk_csgraph, rood_id_csgraph)
    
    if save_raw:
        non_unique_skeleton_ids = sk_df[sk_df.duplicated('skeleton_id', keep=False)]['skeleton_id'].unique()
        filtered_df_unique = sk_df[~sk_df['skeleton_id'].isin(non_unique_skeleton_ids) | (sk_df['syn_id'] == -1)]
        raw_tree = reconstruct_graph(filtered_df_unique, sk_csgraph, rood_id_csgraph)
        with open(file_name, 'wb') as f:
            pickle.dump(raw_tree, f)

    print("Is the reference still a valid tree: ", nx.is_tree(new_tree))
    print("Number of Nodes: ", len(new_tree.nodes))

    return new_tree

def compare(mst, ref_tree):
    '''
    This function compares the two trees and return edit distance
    '''

    ref_tree = reindex_tree_by_attribute(ref_tree, 'syn_id')
    mst = reindex_tree_by_attribute(mst, 'syn_id')

    draw_trees_side_by_side(mst, ref_tree, "MST", "Reference Tree")

    zss_tree1 = build_zss_tree(mst, -1)
    zss_tree2 = build_zss_tree(ref_tree, -1)
    tree_distance = simple_distance(zss_tree1, zss_tree2)
    print("--------------------------------")
    print(f"Tree editing distance: {tree_distance}")

    # edit distance for sequence 
    paths1 = all_paths_to_leaves(mst, -1)
    paths2 = all_paths_to_leaves(ref_tree, -1)
    min_edit_dic1 = min_edit_distance(paths1, paths2)
    if len(min_edit_dic1) != 0:
        val1= sum(min_edit_dic1.values())/(len(min_edit_dic1))
    else:
        val1 = None

    min_edit_dic2 = min_edit_distance(paths2, paths1)
    if len(min_edit_dic2) != 0:
        val2 = sum(min_edit_dic2.values())/(len(min_edit_dic2))
    else:
        val2 = None
    if val1!= None and val2!= None:
        seq_editing = (val1+val2)/2
    else:
        seq_editing = None
    print("Sequence editing distance avg:", seq_editing)

    return len(mst.nodes), tree_distance, seq_editing


def main(mst_path):

    with open(mst_path, 'rb') as f:
        mst = pickle.load(f)
    
    mst, syn_id_wanted, sk_df, sk_csgraph, rood_id_csgraph = prune_mst(mst)
    ref_tree = prune_ref_tree(syn_id_wanted, sk_df, sk_csgraph, rood_id_csgraph)
    number_nodes, tree_distance, seq_editing = compare(mst, ref_tree)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mst_path', type=str, default="data/test_mst.pkl")
    args = parser.parse_args()
    main(args.mst_path)

