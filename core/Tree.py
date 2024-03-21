import numpy as np
import pandas as pd
from copy import deepcopy
import networkx as nx

from core.Branch import BranchSeq

class Tree:

    def __init__(self, graph, root_id=-1):
        '''Directed graphs have edges from leaf to root. 
        The root is the only node with no outgoing edges.
        The leaves are the only nodes with no incoming edges.
        '''
        self.graph = graph
        self.cell_id = graph.graph['cell_id']
        self.shuffle_nodes = [n for n in self.graph.nodes() if self.graph.nodes[n].get('cell_type', 'Unknown') != 'Unknown']
        self.root_id = root_id

    def get_leaves(self):
        '''Return a list of leaf nodes in the graph.'''
        return [node for node in self.graph.nodes() if self.graph.in_degree(node) == 0]
    
    def get_branches(self):
        '''For each leaf node in graph, get the path from the leaf to the root.
        Returns a list of BranchSeq objects, one for each leaf-to-root path.
        '''
        branches = []
        for leaf in self.get_leaves():
            path = nx.shortest_path(self.graph, source=leaf, target=self.root_id)
            branches.append(BranchSeq(path, self.graph, hash((self.cell_id, leaf))))
        return branches
    
    def get_paths(self, duplicate_tail=False):

        graph = self.graph.reverse()

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
    

    def set_coarse_tree(self):

        # Extract the original graph and paths from the tree object
        original_graph = self.graph.reverse()
        paths = self.get_paths(duplicate_tail=True)
        
        # Initialize the new directed graph for the coarse tree
        coarse_tree = nx.DiGraph()
        
        # Helper function to find the upstream head for a given head
        def find_upstream_head(head, paths):
            for path in paths:
                if head in path[1:]:  # Check if the head is somewhere in the tail of another path
                    return path[0]  # Return the head of that path
            return None

        # Process each path
        for path in paths:
            head = path[0]
            
            # Ensure the head node is added to the coarse tree
            if not coarse_tree.has_node(head):
                coarse_tree.add_node(head)
            
            # Find the upstream path head, if it exists
            upstream_head = find_upstream_head(head, paths)
            
            # If there's an upstream head, connect it to the current head in the coarse tree
            if upstream_head and upstream_head != head:  # Ensure we're not adding a self-loop
                if not coarse_tree.has_node(upstream_head):
                    coarse_tree.add_node(upstream_head)
                coarse_tree.add_edge(upstream_head, head)
            
            if len(path) > 1 and len(original_graph.successors(path[-1])) == 0:
                # If the tail node is a leaf node, add it to the coarse tree
                if not coarse_tree.has_node(path[-1]):
                    coarse_tree.add_node(path[-1])
                # Connect the head to the tail in the coarse tree
                coarse_tree.add_edge(head, path[-1])
        
        # Add the root node (-1) and connect it to its successors in the coarse tree
        coarse_tree.add_node(self.root_id)
        # Identify the root's successors in the original graph
        root_successors = list(original_graph.successors(self.root_id))
        for successor in root_successors:
            # Since successors of the root are heads of some paths, they should be in the coarse tree
            if successor in coarse_tree:
                coarse_tree.add_edge(self.root_id, successor)

        self.coarse_tree = coarse_tree
        return coarse_tree
    
    def set_strahler_order(self):
        
        # original_graph = self.graph.reverse()

        for node in self.coarse_tree.nodes:
            if self.coarse_tree.out_degree(node) == 0:  # Leaf node
                self.coarse_tree.nodes[node]['strahler'] = 1 #min(max(1,len(list(original_graph.successors(node)))),2)
        
        # Recursive computation of Strahler numbers
        def assign_strahler(node):
            if 'strahler' in self.coarse_tree.nodes[node]:
                return self.coarse_tree.nodes[node]['strahler']
            
            children_strahler = [assign_strahler(child) for child in self.coarse_tree.successors(node)]
            max_strahler = max(children_strahler)
            if children_strahler.count(max_strahler) > 1:
                self.coarse_tree.nodes[node]['strahler'] = max_strahler + 1
            else:
                self.coarse_tree.nodes[node]['strahler'] = max_strahler
            return self.coarse_tree.nodes[node]['strahler']
        
        # Assuming -1 is the root node
        assign_strahler(self.root_id)

        # return a list of strahler numbers for each node
        return [self.coarse_tree.nodes[node]['strahler'] for node in self.coarse_tree.nodes if node != self.root_id]
    
    def assign_centrifugal_order(self):
        
        for node in self.coarse_tree.nodes:
            dist_from_root = nx.shortest_path_length(self.coarse_tree, source=self.root_id, target=node)
            self.coarse_tree.nodes[node]['centrifugal'] = dist_from_root
        
        return [self.coarse_tree.nodes[node]['centrifugal'] for node in self.coarse_tree.nodes if node != self.root_id]

    def get_random_shuffle(self):
        nodes_to_shuffle, pre_cell_ids, cell_types = zip(*[(n, self.graph.nodes[n]['pre_cell_id'], self.graph.nodes[n]['cell_type']) 
                                                           for n in self.shuffle_nodes])

        permutation = np.random.permutation(len(nodes_to_shuffle))
        
        random_tree = deepcopy(self)
        for i, node in enumerate(nodes_to_shuffle):
            random_tree.graph.nodes[node]['pre_cell_id'] = pre_cell_ids[permutation[i]]
            random_tree.graph.nodes[node]['cell_type'] = cell_types[permutation[i]]

        return random_tree

    def get_type_shuffle(self):

        nodes_to_shuffle, pre_cell_ids, cell_types = zip(*[(n, self.graph.nodes[n]['pre_cell_id'], self.graph.nodes[n]['cell_type']) 
                                                           for n in self.shuffle_nodes])

        synapses_by_type = {}
        for cell_id, cell_type in zip(pre_cell_ids, cell_types):
            if cell_type not in synapses_by_type:
                synapses_by_type[cell_type] = []
            synapses_by_type[cell_type].append(cell_id)

        for cell_type in synapses_by_type:
            np.random.shuffle(synapses_by_type[cell_type])
        
        random_tree = deepcopy(self)
        for i, node in enumerate(nodes_to_shuffle):
            random_tree.graph.nodes[node]['pre_cell_id'] = synapses_by_type[cell_types[i]].pop()

        return random_tree

    def get_axon_shuffle(self, score_mat):
        nodes_to_shuffle, pre_cell_ids, cell_types = zip(*[(n, self.graph.nodes[n]['pre_cell_id'], self.graph.nodes[n]['cell_type']) 
                                                           for n in self.shuffle_nodes])

        if len(nodes_to_shuffle) != score_mat.shape[0]:
            raise ValueError(f'Score matrix must have the same number of rows ({score_mat.shape[0]}) as shuffling-nodes in the tree ({len(nodes_to_shuffle)}).')
        
        random_tree = deepcopy(self)
        permutation = BranchSeq.sample_permutation(np.array(score_mat).copy())
        for i, node in enumerate(nodes_to_shuffle):
            random_tree.graph.nodes[node]['pre_cell_id'] = pre_cell_ids[permutation[i]]
            random_tree.graph.nodes[node]['cell_type'] = cell_types[permutation[i]]
        
        return random_tree
