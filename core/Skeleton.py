import numpy as np
import pandas as pd
from copy import deepcopy
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import breadth_first_order
from scipy.sparse import csr_matrix
import pickle


class Skeleton:

    @staticmethod
    def connect_disjoint_branches(G, soma_node=-1):
        G = deepcopy(G)
        G.remove_edges_from(list(nx.selfloop_edges(G)))

        components = list(nx.connected_components(G))
        root_component = [c for c in components if soma_node in c][0]
        components.remove(root_component)

        soma_pos = G.nodes[soma_node]['pos']

        for component in components:
            component = list(component)
            leaf_nodes = [node for node in component if G.degree(node) <= 1]
            # if len(leaf_nodes) == 0:
            #     print(component, [G.degree(node) for node in component])
                
            distances = [np.linalg.norm(G.nodes[leaf_node]['pos'] - soma_pos) for leaf_node in leaf_nodes]

            closest_leaf_node = leaf_nodes[np.argmin(distances)]
            G.add_edge(soma_node, closest_leaf_node)
    
        return G
    
    @staticmethod
    def direct_tree_from_root(mst, soma_node=-1):
        directed_tree = nx.DiGraph(cell_id=mst.graph['cell_id'], cell_type=mst.graph['cell_type'])

        # Copy node attributes
        for node, attrs in mst.nodes(data=True):
            directed_tree.add_node(node, **attrs)

        visited = {node: False for node in mst.nodes()}
        
        queue = [soma_node]
        visited[soma_node] = True
        
        while queue:
            node = queue.pop(0)
            
            for neighbor in mst.neighbors(node):
                if not visited[neighbor]:
                    # Copy the edge attributes
                    directed_tree.add_edge(node, neighbor, **mst[node][neighbor])
                    
                    visited[neighbor] = True
                    queue.append(neighbor)
                    
        return directed_tree
    
    @staticmethod
    def is_start_of_twig(node, tree):
        """Check if the node is the start of a twig."""
        parents = list(tree.predecessors(node))
        siblings = [child for parent in parents for child in tree.successors(parent) if child != node]
        
        # A node is the start of a twig if its parent has multiple children
        return bool(siblings)

    @staticmethod
    def walk_twig(node, tree):
        """Walk the twig starting from the node and return the nodes in the twig."""
        twig_nodes = [node]
        children = list(tree.successors(node))
        
        if len(children) > 1:
            return [node, node]

        while len(children) == 1:
            node = children[0]
            twig_nodes.append(node)
            children = list(tree.successors(node))
        
        return twig_nodes

    @staticmethod
    def closest_sibling(node, siblings, G):
        dists = [np.linalg.norm(G.nodes[node]['pos'] - G.nodes[sib]['pos']) for sib in siblings]
        return siblings[np.argmin(dists)]

    @staticmethod
    def merge_twig(twig_nodes, sibling, parent, G, log=False):
        sibling_twig_nodes = Skeleton.walk_twig(sibling, G)
        if len(sibling_twig_nodes)>1 and sibling_twig_nodes[0] == sibling_twig_nodes[1]:
            sibling_twig_nodes = [sibling]

        # calculate distances from parent to all nodes in twig and sibling twig
        all_nodes = np.array(twig_nodes + sibling_twig_nodes)
        all_dists = [np.linalg.norm(G.nodes[node]['pos'] - G.nodes[parent]['pos']) for node in twig_nodes]
        all_dists.extend([np.linalg.norm(G.nodes[node]['pos'] - G.nodes[parent]['pos']) for node in sibling_twig_nodes])

        # reorder nodes in twig and sibling twig based on distance to parent
        ordered_nodes = all_nodes[np.argsort(all_dists)]

        # print("Merging twig nodes {} and sibling twig nodes {} to {}".format(twig_nodes, sibling_twig_nodes, ordered_nodes))
        
        # remove edges in twig, sibling twig, and from both to parent
        if len(twig_nodes) > 1:
            for pnode, cnode in zip(twig_nodes[:-1], twig_nodes[1:]):
                G.remove_edge(pnode, cnode)
        if len(sibling_twig_nodes) > 1:
            for pnode, cnode in zip(sibling_twig_nodes[:-1], sibling_twig_nodes[1:]):
                G.remove_edge(pnode, cnode)
        G.remove_edge(parent, twig_nodes[0])
        G.remove_edge(parent, sibling_twig_nodes[0])

        # add edges based on ordered nodes
        G.add_edge(parent, ordered_nodes[0], weight=np.linalg.norm(G.nodes[ordered_nodes[0]]['pos'] - G.nodes[parent]['pos']))
        for pnode, cnode in zip(ordered_nodes[:-1], ordered_nodes[1:]):
            G.add_edge(pnode, cnode, weight=np.linalg.norm(G.nodes[cnode]['pos'] - G.nodes[pnode]['pos']))
        
        return
    
    @staticmethod
    def remove_short_twigs(tree, k):
        while True:
            twig_to_remove = None

            # 1. Identify a short twig
            for node in tree.nodes():
                if len(list(tree.successors(node))) == 0:
                    siblings = [child for parent in tree.predecessors(node) for child in tree.successors(parent) if child != node]
                    if len(siblings) > 0:
                        twig_to_remove = ([node], Skeleton.closest_sibling(node, siblings, tree))
                elif Skeleton.is_start_of_twig(node, tree):
                    twig_nodes = Skeleton.walk_twig(node, tree)
                    if len(twig_nodes) > k:
                        continue
                    elif len(twig_nodes) == 2 and twig_nodes[0] == twig_nodes[1]:
                        continue
                    else:
                        start_node = twig_nodes[0]
                        siblings = [child for parent in tree.predecessors(start_node) 
                                    for child in tree.successors(parent) if child != start_node]
                        
                        # Check if there are sibling twigs
                        if any(Skeleton.is_start_of_twig(sibling, tree) for sibling in siblings):
                            twig_to_remove = (twig_nodes, Skeleton.closest_sibling(start_node, siblings, tree))
                            break
            
            # If no twigs to remove, exit loop
            if not twig_to_remove:
                break

            # 2. Merge the twig
            parent = list(tree.predecessors(twig_to_remove[0][0]))[0]
            Skeleton.merge_twig(*twig_to_remove, parent, tree, log=True)
                        
        return tree
    
    @staticmethod
    def prune_unknown_twigs(tree, pre=True):

        leaves = [node for node, degree in tree.out_degree() if degree == 0]

        try:
            if tree.nodes[leaves[0]]['cell_type']:
                key = 'cell_type'
        except KeyError:
            key = 'cell_type_pre' if pre else 'cell_type_post'

        # get the path-length from each leaf to its first parent/grandparent/grandgrandparent/... node with a sibling
        # if no node with a sibling is found, the path length is the length of the path to the root
        inh_paths = []
        for leaf in leaves:
            l = 1
            path = [deepcopy(leaf)]
            siblings = [child for parent in tree.predecessors(leaf) for child in tree.successors(parent) if child != leaf]
            while len(siblings) == 0 and l < len(tree.nodes()):
                l += 1
                leaf = list(tree.predecessors(leaf))[0]
                path.append(deepcopy(leaf))
                siblings = [child for parent in tree.predecessors(leaf) for child in tree.successors(parent) if child != leaf]

            if np.all([tree.nodes[node][key]=='Unknown' for node in path]):
                inh_paths.append(path)

        # remove all pure unknown twigs
        for path in inh_paths:
            for node in path:
                tree.remove_node(node)

        return tree
    
    @staticmethod
    def extract_paths_from_graph(graph, duplicate_tail=False):

        # first get all leaves
        leaves = [node for node, degree in graph.out_degree() if degree == 0]

        # for each leaf, walk to its parent until you reach a node with no parents
        # save a list of (node, number of children) tuples
        paths = set()
        for leaf in leaves:
            path = [(leaf, 0)]
            current_node = leaf
            while len(list(graph.predecessors(current_node))) > 0:
                parent = list(graph.predecessors(current_node))[0]
                num_siblings = len(list(graph.successors(parent)))
                path.append((parent, num_siblings))
                current_node = parent

            # reverse the path so that it goes from soma to leaf, then skip soma node
            path.reverse()
            path = path[1:]

            # split the path at nodes with more than one child, duplicating the node in both lists
            split_paths = []
            current_path = []
            for node, num_children in path:
                current_path.append(node)
                if num_children > 1 or num_children == 0:
                    split_paths.append(tuple(current_path))
                    if duplicate_tail:
                        current_path = [node]
                    else:
                        current_path = []

            paths.update(split_paths)
        
        # remove paths that are empty
        paths = [path for path in paths if len(path) > 0]
        return paths
    
    @staticmethod
    def prune_graph_with_synapses(df, csgraph, root_id):
        """
        This function is used to prune the csgraph created by the skeleton and convert to networkx tree
        What synapses to keep is decided by the input argument df
        Note, for instances where one node corresponds to multiple synapses, we choose one of those synapses and ignore the remaining

        @param pandas dataframe df: The dataframe of synapses to keep, including skeleton_id and syn_id
        @param csgraph csgraph: Unpruned tree
        @param int root_id: The skeleton_id corresponding to the soma
        @return networkx tree new_tree: The pruned tree, with syn_id as attribute
        """

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


    def __init__(self, cell_info, syn_group, syn_k=6, soma_k=12):
        
        self.cell_id = cell_info['pt_root_id'].values[0]
        self.cell_type = cell_info['cell_type'].values[0]
        self.n_synapses = syn_group.shape[0]

        group_size = syn_group.shape[0]
        if group_size < syn_k:
            syn_k = group_size
        if group_size < soma_k:
            soma_k = group_size
        
        self.syn_k = syn_k
        self.soma_k = soma_k

        self.mst = self.skeletonize(cell_info, syn_group, syn_k, soma_k)
        self.smooth_mst = None
        self.emst = None

    def skeletonize(self, cell_info, syn_group, syn_k, soma_k):

        # Keep relevant rows of synapse table
        synapses = syn_group[['ctr_pt_x', 'ctr_pt_y', 'ctr_pt_z']]

        # Get the soma location for the cell
        # cell_info = cells_df.loc[cell_id]
        cell_id = cell_info['pt_root_id'].values[0]
        cell_type = cell_info['cell_type'].values[0]
        soma_xyz = np.array(cell_info[['pt_x', 'pt_y', 'pt_z']].values)
        # soma_xyz = np.matmul(soma_xyz, np.diag([4/1000, 4/1000, 40/1000]))

        # Add the soma location to the synapse table
        soma_df = pd.DataFrame(soma_xyz)
        soma_df.columns = ['ctr_pt_x', 'ctr_pt_y', 'ctr_pt_z']
        soma_df.index = [-1]
        synapses_w_soma = pd.concat([synapses, soma_df])

        # Create a kdtree from the synapse locations
        kd_tree = NearestNeighbors(n_neighbors=syn_k, algorithm='kd_tree').fit(synapses_w_soma.values)

        # Get the k nearest neighbors for each synapse and the soma
        distances, indices = kd_tree.kneighbors(synapses.values)
        soma_distances, soma_indices = kd_tree.kneighbors(soma_xyz.reshape(1, -1), n_neighbors=soma_k)

        # Subtract the "radius" of the soma from the soma distances
        if soma_distances.shape[1] > 1:
            soma_radius = soma_distances[0][1]
        else:
            soma_radius = 3
        soma_distances = soma_distances - soma_radius 

        # Create a graph from the synapse group
        nodes = list(synapses.index.values)
        G = nx.Graph(cell_id=cell_id, cell_type=cell_type)
        for node in nodes:
            node_ct_pre = syn_group.loc[node, 'cell_type_pre']
            node_ct_post = syn_group.loc[node, 'cell_type_post']
            node_id = syn_group.loc[node, 'id']
            node_pre_id = syn_group.loc[node, 'pre_pt_root_id']
            node_post_id = syn_group.loc[node, 'post_pt_root_id']
            G.add_node(node, pos=synapses.loc[node, ['ctr_pt_x', 'ctr_pt_y', 'ctr_pt_z']].values,
                             cell_type_pre=node_ct_pre,
                             cell_type_post=node_ct_post,
                             syn_id=node_id,
                             pre_cell_id=node_pre_id,
                             post_cell_id=node_post_id)
        nodes.append(-1)
        G.add_node(-1, pos=soma_xyz, cell_type_pre=-1, cell_type_post=-1, syn_id=-1, pre_cell_id=-1, post_cell_id=-1)

        # Add edges according to the kdtree
        for i in range(len(indices)):
            syn_id = nodes[i]
            for j in range(len(indices[i])):
                if i != indices[i][j]:
                    G.add_edge(syn_id, nodes[indices[i][j]], weight=distances[i][j])
        
        # Add edges from the soma to its nearest neighbors, corrected for the radius of the soma
        # Make sure not to add an edge from the soma to itself
        for l in range(1,len(soma_indices[0])):
            G.add_edge(-1, nodes[soma_indices[0][l]], weight=soma_distances[0][l])

        # Get the minimum spanning tree
        mst = nx.minimum_spanning_tree(G)

        # Make the graph fully connected
        mst = Skeleton.connect_disjoint_branches(mst)

        # Direct the tree from the soma
        mst = Skeleton.direct_tree_from_root(mst, soma_node=-1)

        return mst

    def smooth(self, twig_length, prune_unknown=True):
        DG = deepcopy(self.mst)
        if prune_unknown:
            DG = Skeleton.prune_unknown_twigs(DG)
        
        DG = Skeleton.remove_short_twigs(DG, twig_length)
        self.smooth_mst = DG
        self.twig_length = twig_length

        return self.smooth_mst
    
    def extract_excitatory_smooth_mst(self, pre=True):
        DG = deepcopy(self.smooth_mst)

        try:
            if DG.nodes[0]['cell_type']:
                key = 'cell_type'
        except KeyError:
            key = 'cell_type_pre' if pre else 'cell_type_post'
        
        nodes_to_remove = []
        for node in DG.nodes:
            if node > 0 and DG.nodes[node][key] == 'Unknown' and len(list(DG.successors(node))) < 2:
                nodes_to_remove.append(node)
        for node in nodes_to_remove:
            children = list(DG.successors(node))
            parent = list(DG.predecessors(node))[0]
            DG.remove_node(node)
            for child in children:
                DG.add_edge(parent, child)
        
        self.emst = DG
        return DG
    
    def get_paths(self, smoothed=True, exc=True, duplicate_tail=False):

        if exc:
            graph = self.emst
        elif smoothed:
            graph = self.smooth_mst
        else:
            graph = self.mst

        # first get all leaves
        leaves = [node for node, degree in graph.out_degree() if degree == 0]

        # for each leaf, walk to its parent until you reach a node with no parents
        # save a list of (node, number of children) tuples
        paths = set()
        for leaf in leaves:
            path = [(leaf, 0)]
            current_node = leaf
            while len(list(graph.predecessors(current_node))) > 0:
                parent = list(graph.predecessors(current_node))[0]
                num_siblings = len(list(graph.successors(parent)))
                path.append((parent, num_siblings))
                current_node = parent

            # reverse the path so that it goes from soma to leaf, then skip soma node
            path.reverse()
            path = path[1:]

            # split the path at nodes with more than one child, duplicating the node in both lists
            split_paths = []
            current_path = []
            for node, num_children in path:
                current_path.append(node)
                if num_children > 1 or num_children == 0:
                    split_paths.append(tuple(current_path))
                    if duplicate_tail:
                        current_path = [node]
                    else:
                        current_path = []

            paths.update(split_paths)
        
        return paths