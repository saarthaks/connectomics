import numpy as np
from numba import jit
import networkx as nx
from copy import deepcopy
import pylcs as LCS

class BranchSeq:

    @staticmethod
    def collapse_path(path, graph):
        if not path:
            return []

        new_seq = [path[0]]  # Start with the first node of the sequence

        for i in range(1, len(path)):
            current_node = path[i]
            previous_node = new_seq[-1]

            # Check if the current node has the same pre_cell_id as the previous node
            if graph.nodes[current_node].get('pre_cell_id') != graph.nodes[previous_node].get('pre_cell_id'):
                new_seq.append(current_node)
            elif i == len(path) - 1:
                # If it's the last node, add it regardless of the pre_cell_id
                new_seq.append(current_node)

        return new_seq
    
    @staticmethod
    def sequence_to_string(sequence, pt_root_to_char_dict=None):
        if pt_root_to_char_dict is None:
            return sequence
        
        return ''.join([pt_root_to_char_dict[node] for node in sequence])
    
    @staticmethod
    def lcs_list(seq1, other_seqs):
        return LCS.lcs_sequence_of_list(seq1, other_seqs)
    
    @staticmethod
    def lcs_dist(branch1, branch2, pt_root_to_char_dict):
        seq1 = branch1.get_sequence(pt_root_to_char_dict=pt_root_to_char_dict)
        seq2 = branch2.get_sequence(pt_root_to_char_dict=pt_root_to_char_dict)

        seq_1_idx = list(filter(lambda i: i != -1, LCS.lcs_sequence_idx(seq2, seq1)))
        min_1, max_1 = min(seq_1_idx), max(seq_1_idx)
        dist_1 = np.linalg.norm(branch1.syn_pos_sequence['collapsed'][min_1] - branch1.syn_pos_sequence['collapsed'][max_1])
        
        seq_2_idx = list(filter(lambda i: i != -1, LCS.lcs_sequence_idx(seq1, seq2)))
        min_2, max_2 = min(seq_2_idx), max(seq_2_idx)
        dist_2 = np.linalg.norm(branch2.syn_pos_sequence['collapsed'][min_2] - branch2.syn_pos_sequence['collapsed'][max_2])

        return dist_1, dist_2

    @staticmethod
    def lcs_dist_list(branch, other_branches, pt_root_to_char_dict):
        seq1 = branch.get_sequence(pt_root_to_char_dict=pt_root_to_char_dict)
        other_seqs = [other_branch.get_sequence(pt_root_to_char_dict=pt_root_to_char_dict) for other_branch in other_branches]
        all_lcs = BranchSeq.lcs_list(seq1, other_seqs)
        dists = []
        for lcs, branch2 in zip(all_lcs, other_branches):
            if lcs > 1:
                dists.append((lcs, BranchSeq.lcs_dist(branch, branch2, pt_root_to_char_dict)))
            else:
                dists.append((lcs, (np.inf, np.inf)))
        
        return dists
    
    @staticmethod
    @jit(nopython=True)
    def sample_permutation(score_matrix):
        L = score_matrix.shape[0]
        chosen_rows = np.arange(L)
        chosen_columns = np.empty(L, dtype=np.int32)
        for i in range(L):
            # Normalize scores for the chosen row, excluding already chosen columns
            row = chosen_rows[i]
            probabilities = score_matrix[row, :]
            if i > 0:
                # Set probabilities of already chosen columns to 0
                probabilities[chosen_columns[:i]] = 0
            probabilities /= probabilities.sum()

            # Choose a column based on the normalized probabilities
            column = np.searchsorted(np.cumsum(probabilities), np.random.random(), side="right")
            chosen_columns[i] = column

        return chosen_columns
    
    def __init__(self, path, graph, id):

        self.cell_id = graph.graph['cell_id']
        self.branch_id = id
        self.syn_id_sequence = {'raw': [], 'collapsed': []}
        self.syn_pos_sequence = {'raw': [], 'collapsed': []}
        self.cell_id_sequence = {'raw': [], 'collapsed': []}
        self.cell_type_sequence = {'raw': [], 'collapsed': []}

        try:
            if path[0]['cell_type']:
                cell_type_key = 'cell_type'
        except KeyError:
            cell_type_key = 'cell_type_pre'
        
        for node in path:
            self.syn_id_sequence['raw'].append(graph.nodes[node]['syn_id'])
            self.syn_pos_sequence['raw'].append(graph.nodes[node]['pos'])
            self.cell_id_sequence['raw'].append(graph.nodes[node]['pre_cell_id'])
            self.cell_type_sequence['raw'].append(graph.nodes[node][cell_type_key])
        
        collapsed_path = BranchSeq.collapse_path(path, graph)
        for node in collapsed_path:
            self.syn_id_sequence['collapsed'].append(graph.nodes[node]['syn_id'])
            self.syn_pos_sequence['collapsed'].append(graph.nodes[node]['pos'])
            self.cell_id_sequence['collapsed'].append(graph.nodes[node]['pre_cell_id'])
            self.cell_type_sequence['collapsed'].append(graph.nodes[node][cell_type_key])
    
    def length(self, collapsed=True):
        if collapsed:
            return len(self.syn_id_sequence['collapsed'])
        else:
            return len(self.syn_id_sequence['raw'])
    
    def get_sequence(self, collapsed=True, pt_root_to_char_dict=None):
        if collapsed:
            return BranchSeq.sequence_to_string(self.syn_id_sequence['collapsed'], pt_root_to_char_dict)
        else:
            return BranchSeq.sequence_to_string(self.syn_id_sequence['raw'], pt_root_to_char_dict)

    def distance(self, collapsed=True):
        if collapsed:
            return np.linalg.norm(self.syn_pos_sequence['collapsed'][0] - self.syn_pos_sequence['collapsed'][-1])
        else:
            return np.linalg.norm(self.syn_pos_sequence['raw'][0] - self.syn_pos_sequence['raw'][-1])
        
    def get_random_shuffle(self, collapsed=True):
        if not collapsed:
            return ValueError("Can only shuffle collapsed sequences")
        
        permutation = np.random.permutation(len(self.syn_id_sequence['collapsed']))
        random_branch = deepcopy(self)
        random_branch.syn_id_sequence['collapsed'] = [self.syn_id_sequence['collapsed'][i] for i in permutation]
        random_branch.cell_id_sequence['collapsed'] = [self.cell_id_sequence['collapsed'][i] for i in permutation]
        random_branch.cell_type_sequence['collapsed'] = [self.cell_type_sequence['collapsed'][i] for i in permutation]
        return random_branch
    
    def get_type_shuffle(self, collapsed=True):
        if not collapsed:
            return ValueError("Can only shuffle collapsed sequences")
        
        synapses_by_type = {}
        for cell_id, syn_id, cell_type in zip(self.cell_id_sequence['collapsed'], self.syn_id_sequence['collapsed'], self.cell_type_sequence['collapsed']):
            if cell_type not in synapses_by_type:
                synapses_by_type[cell_type] = []
            synapses_by_type[cell_type].append((cell_id, syn_id))

        for cell_type in synapses_by_type:
            np.random.shuffle(synapses_by_type[cell_type])
        
        random_branch = deepcopy(self)
        new_syn_id_sequence = []
        new_cell_id_sequence = []
        for cell_type in random_branch.cell_type_sequence['collapsed']:
            cell_id, syn_id = synapses_by_type[cell_type].pop()
            new_syn_id_sequence.append(syn_id)
            new_cell_id_sequence.append(cell_id)
        
        random_branch.syn_id_sequence['collapsed'] = new_syn_id_sequence
        random_branch.cell_id_sequence['collapsed'] = new_cell_id_sequence
        return random_branch

    def get_axon_shuffle(self, score_mat, collapsed=True):
        if not collapsed:
            return ValueError("Can only shuffle collapsed sequences")
        
        perm = BranchSeq.sample_permutation(np.array(score_mat).copy())
        random_branch = deepcopy(self)
        random_branch.syn_id_sequence['collapsed'] = [self.syn_id_sequence['collapsed'][i] for i in perm]
        random_branch.cell_id_sequence['collapsed'] = [self.cell_id_sequence['collapsed'][i] for i in perm]
        random_branch.cell_type_sequence['collapsed'] = [self.cell_type_sequence['collapsed'][i] for i in perm]
        return random_branch