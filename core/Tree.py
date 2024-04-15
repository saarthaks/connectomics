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
        self.root_id = root_id

    def get_leaves(self):
        '''Return a list of leaf nodes in the graph.'''
        return [node for node in self.graph.nodes() if self.graph.in_degree(node) == 0]

    def get_paths(self, duplicate_tail=False):

        graph = self.graph.reverse()

        # first get all leaves
        leaves = [n for n in graph.nodes if graph.out_degree(n)==0]

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
