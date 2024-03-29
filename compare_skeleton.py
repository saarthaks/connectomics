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
import pandas as pd
from core.Skeleton import Skeleton


class Node:
    """
    This class is used for converting a networkx tree to a zss tree
    for tree editing distance calculation
    """

    def __init__(self, label):
        self.label = label
        self.children = []
        
    @classmethod
    def build_zss_tree(cls, nx_tree, root_label):
        root = cls(root_label)
        cls._build_tree_helper(nx_tree, root, root_label, set())
        return root

    @staticmethod
    def _build_tree_helper(nx_tree, current_node, current_label, visited):
        visited.add(current_label)
        children = []
        for neighbor in nx_tree.neighbors(current_label):
            if neighbor not in visited:
                child_node = Node(neighbor)
                children.append(child_node)
                Node._build_tree_helper(nx_tree, child_node, neighbor, visited)
        current_node.children = sorted(children, key=lambda x: x.label)


def reindex_tree_by_attribute(tree, attribute='syn_id'):
    """
    This function reindex the tree with syn_id for easier comparision
    """

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
    """
    This function draws the two trees side by side, with an option to save the plot

    @param networkx graph tree1: The first tree we want to draw
    @param networkx graph tree2: The second tree we want to draw
    @param str title1: The title of the first tree
    @param str title2: The title of the second tree
    @param str path: The path the save the plot, default to None
    """

    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    pos = nx.spring_layout(tree1)  
    nx.draw(tree1, pos, with_labels=True, ax=axs[0], node_size=200, node_color='skyblue',  font_size = 6)
    axs[0].set_title(title1)

    pos = nx.spring_layout(tree2)
    nx.draw(tree2, pos, with_labels=True, ax=axs[1], node_size=200, node_color='lightgreen', font_size = 6)
    axs[1].set_title(title2)

    if path:
        plt.savefig(path)
    plt.show()


def convert_to_cs(tree):
    """
    This function is used to convert a networkx tree to csgraph tree

    @param networkx graph tree: The MST we want to prune
    @return: csgraph adj_matrix: The csgraph of the tree
    @return: int root_id: The skeleton_id of the root node (soma)
    @return: pandas dataframe df: The dataframe of the skeleton, including skeleton_id and syn_id
    """
    
    node_data = []
    for node in tree.nodes(data=True):
        node_id = node[0] 
        syn_id = node[1]['syn_id'] 
        node_data.append({'skeleton_id': int(node_id), 'syn_id': int(syn_id)})
        
    df = pd.DataFrame(node_data)
    root_node = df[df['syn_id'] == -1]['skeleton_id'].iloc[0]
    new_id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(df['skeleton_id'], key=lambda x: x == root_node))}

    df['skeleton_id'] = df['skeleton_id'].apply(lambda x: new_id_mapping[x])
    tree = nx.relabel_nodes(tree, new_id_mapping)
    root_id = df[df['syn_id'] == -1]['skeleton_id'].iloc[0]
    adj_matrix = nx.to_scipy_sparse_matrix(tree, nodelist=sorted(tree.nodes()), weight=None, dtype=int)
    adj_matrix.indices = adj_matrix.indices.astype(np.int32)
    adj_matrix.indptr = adj_matrix.indptr.astype(np.int32)

    return adj_matrix, root_id, df


def skeleton_syn_to_keep(cell_table_path, mst, cell_df=None):
    """
    This function creates the reference skeleton 
    and decides what synapse to keep based on the mst (usually only excitatory)
    Note, if more than one synapses have the same skeleton_id, discard them all

    @param str cell_table_path: The path the reference cell table
    @param networkx tree mst: The MST we want to construct reference tree to
    @return list syn_id_wanted: The list of syn_id that we want to keep in the reference tree
    @return pandas df sk_df: The dataframe of the pruned skeleton, including skeleton_id and syn_id
    @return csgraph sk_csgraph: The csgraph of the skeleton
    @return int rood_id_csgraph: The skeleton_id of the root node (soma)
    """

    example_cell_id = int(mst.graph["cell_id"])
    print(example_cell_id)
    client = CAVE()
    cell_df = pd.read_csv(cell_table_path)
    sk_df, sk_csgraph, rood_id_csgraph = client.download_sk_anno(example_cell_id, cell_df)

    syn_id_wanted = []
    non_unique_skeleton_ids = sk_df[sk_df.duplicated('skeleton_id', keep=False)]['skeleton_id'].unique()
    # only synapses with unique skeleton_id are kept
    filtered_df_unique = sk_df[~sk_df['skeleton_id'].isin(non_unique_skeleton_ids) | (sk_df['syn_id'] == -1)]

    kept_syn_ids = list(filtered_df_unique["syn_id"])
    for node in mst.nodes(data=True):
        _, attrs = node
        if 'syn_id' in attrs and attrs['syn_id'] in kept_syn_ids:
            syn_id_wanted.append(attrs['syn_id'])

    return syn_id_wanted, sk_df, sk_csgraph, rood_id_csgraph

def prune_tree(syn_id_wanted, sk_df, sk_csgraph, rood_id_csgraph, file_name = None):
    """
    This funciton prunes the csgraph tree 
    with an option to save the raw tree (tree with all synapses, not just excitatory)

    @param list syn_id_wanted: The list of synapes ids we want to keep
    @param pandas df sk_df: The dataframe of the pruned skeleton, including skeleton_id and syn_id
    @param csgraph sk_csgraph: The csgraph of the skeleton
    @param int rood_id_csgraph: The skeleton_id corresponding to the soma
    @param str file_name: The file name where we save the raw trees to. Optinal, default to not saving (None)
    @return networkx tree new_tree: The pruned tree, with syn_id as attribute
    """

    filtered_df = sk_df[sk_df['syn_id'].isin(syn_id_wanted)]

    new_tree = Skeleton.prune_graph_with_synapses(filtered_df, sk_csgraph, rood_id_csgraph)
    
    if file_name:
        non_unique_skeleton_ids = sk_df[sk_df.duplicated('skeleton_id', keep=False)]['skeleton_id'].unique()
        filtered_df_unique = sk_df[~sk_df['skeleton_id'].isin(non_unique_skeleton_ids) | (sk_df['syn_id'] == -1)]
        raw_tree = Skeleton.prune_graph_with_synapses(filtered_df_unique, sk_csgraph, rood_id_csgraph)
        with open(file_name, 'wb') as f:
            pickle.dump(raw_tree, f)

    print("Is the graph still a valid tree: ", nx.is_tree(new_tree))
    print("Number of Nodes: ", len(new_tree.nodes))

    return new_tree

def compare(mst, ref_tree):
    """
    This function compares the two trees and calculates edit distance
    Draws the two trees side by side

    @param networkx tree mst: Pruned mst
    @param networkx tree ref_tree: Pruned reference tree
    @return int n_nodes: The number of nodes in each tree
    @return float tree_distance: The tree editing distance between the two trees
    @return float seq_editing: The average of the ration of sequence editing distance and sequence length (not counting soma)
    """

    ref_tree = reindex_tree_by_attribute(ref_tree, 'syn_id')
    mst = reindex_tree_by_attribute(mst, 'syn_id')
    n_nodes =  len(mst.nodes)

    draw_trees_side_by_side(mst, ref_tree, "MST", "Reference Tree")

    zss_tree1 = Node.build_zss_tree(mst, -1)
    zss_tree2 = Node.build_zss_tree(ref_tree, -1)
    tree_distance = simple_distance(zss_tree1, zss_tree2)
    print("--------------------------------")
    print(f"Tree editing distance: {tree_distance}")

    def all_paths_to_leaves(tree, root):
        paths = []
        for leaf in [x for x in tree.nodes() if tree.degree(x) == 1 and x != root]:
            paths.append(nx.shortest_path(tree, root, leaf))
        return paths
    
    paths1 = all_paths_to_leaves(mst, -1)
    paths2 = all_paths_to_leaves(ref_tree, -1)

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

    return n_nodes, tree_distance, seq_editing


def main(cell_table_path, mst_path):

    with open(mst_path, 'rb') as f:
        mst = pickle.load(f)
    
    syn_id_wanted, sk_df, sk_csgraph, ref_rood_id = skeleton_syn_to_keep(cell_table_path, mst, cell_df=None)
    mst_csgraph, mst_root_id, mst_df = convert_to_cs(mst)
    new_mst = prune_tree(syn_id_wanted, mst_df, mst_csgraph, mst_root_id)
    ref_tree = prune_tree(syn_id_wanted, sk_df, sk_csgraph, ref_rood_id)
    number_nodes, tree_distance, seq_editing = compare(new_mst, ref_tree)

    return number_nodes, tree_distance, seq_editing


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell_table_path', type=str, default="data/cells_no_repeats.csv")
    parser.add_argument('--mst_path', type=str, default="data/test_mst.pkl")
    args = parser.parse_args()
    main(args.mst_path)

