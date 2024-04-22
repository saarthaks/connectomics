import numpy as np
from numba import jit
import networkx as nx
from copy import deepcopy
from collections import defaultdict
import pylcs as LCS
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform

from numba_stats import truncnorm

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

class GenericBranchSeq:

    @staticmethod
    def get_shuffle(all_branches, shuffle_type='random', score_dict=None, update_position=False):
        if shuffle_type == 'original':
            return all_branches

        shuffled_branches = {}
        if shuffle_type == 'random':
            for cell_id, branches in tqdm(all_branches.items()):
                shuffled_branches[cell_id] = [branch.get_random_shuffle() for branch in branches]
        elif shuffle_type == 'type':
            for cell_id, branches in tqdm(all_branches.items()):
                shuffled_branches[cell_id] = [branch.get_type_shuffle() for branch in branches]
        elif shuffle_type == 'axon':
            if score_dict is None:
                raise ValueError('Score matrix not provided')
            for cell_id, branches in tqdm(all_branches.items()):
                if len(score_dict[cell_id]) != len(branches):
                    raise ValueError('Score matrix length does not match branch length')
                shuffled_branches[cell_id] = [branch.get_axon_shuffle(score_mat) for branch, score_mat in zip(branches, score_dict[cell_id])]
        elif shuffle_type == 'continuous':
            if score_dict is None:
                raise ValueError('Score matrix not provided')
            for cell_id, branches in tqdm(all_branches.items()):
                shuffled_branches[cell_id] = [branch.get_continuous_shuffle(prob_dicts, edit_position=update_position) for branch, prob_dicts in zip(branches, score_dict[cell_id])]
        else:
            raise ValueError('Invalid shuffle type')
        
        return shuffled_branches

    @staticmethod
    def make_id2char_dict(pt_root_ids):
        # Printable unicode blocks:
        # 19968-40959: CJK Unified Ideographs
        coding_block_1 = np.arange(19968, 40960, 1)

        # 44032-55215: Hangul Syllables
        coding_block_2 = np.arange(44032, 55216, 1)

        # 131072-173791: CJK Unified Ideographs Extension B
        coding_block_3 = np.arange(131072, 173792, 1)
        code_points = np.concatenate([coding_block_1, coding_block_2, coding_block_3])
        
        pt_root_id_to_char = defaultdict(str)
        char_to_pt_root_id = defaultdict(int)
        for i, pt_root_id in enumerate(pt_root_ids):
            pt_root_id_to_char[pt_root_id] = chr(code_points[i])

        for k, v in pt_root_id_to_char.items():
            char_to_pt_root_id[v] = k
        return pt_root_id_to_char, char_to_pt_root_id

    @staticmethod
    def collapse_path(path, graph):
        if not path or len(path) == 0:
            return []
    
        # filter out nodes with no pre_cell_id
        path = list(filter(lambda node: graph.nodes[node].get('pre_cell_id', -999) != -999, path))

        if len(path) <= 1:
            return path

        start_id = graph.nodes[path[0]]['pre_cell_id']
        if all(graph.nodes[node]['pre_cell_id'] == start_id for node in path):
            return [path[0]]
    
        if len(path) == 2:
            return path

        end_id = graph.nodes[path[-1]]['pre_cell_id']
        if len(path) == 3:
            if start_id == graph.nodes[path[1]]['pre_cell_id']:
                return [path[1], path[2]]
            elif end_id == graph.nodes[path[1]]['pre_cell_id']:
                return [path[0], path[1]]
            else:
                return path
        
        # finds the index of b (start_idx) in ['a', 'a', ..., 'a', 'b', ...]
        for i in range(1, len(path)):
            if graph.nodes[path[i]]['pre_cell_id'] != start_id:
                start_idx = i
                break
        
        new_seq = [path[i-1], path[i]]
        if start_idx == len(path) - 1:
            return new_seq

        # finds the index of b (end_idx) in [..., 'b', 'a', ..., 'a', 'a']
        for j in range(len(path) - 2, -1, -1):
            if graph.nodes[path[j]]['pre_cell_id'] != end_id:
                end_idx = j
                break
        
        
        for k in range(start_idx + 1, end_idx + 2):
            current_node = path[k]
            previous_node = new_seq[-1]

            # Check if the current node has the same pre_cell_id as the previous node
            if graph.nodes[current_node]['pre_cell_id'] != graph.nodes[previous_node]['pre_cell_id']:
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
        subseq1 = branch1.subsequence(seq_1_idx)
        # min_1, max_1 = min(seq_1_idx), max(seq_1_idx)
        # dist_1 = np.linalg.norm(branch1.syn_pos_sequence['collapsed'][min_1] - branch1.syn_pos_sequence['collapsed'][max_1])
        
        seq_2_idx = list(filter(lambda i: i != -1, LCS.lcs_sequence_idx(seq1, seq2)))
        subseq2 = branch2.subsequence(seq_2_idx)
        # min_2, max_2 = min(seq_2_idx), max(seq_2_idx)
        # dist_2 = np.linalg.norm(branch2.syn_pos_sequence['collapsed'][min_2] - branch2.syn_pos_sequence['collapsed'][max_2])

        # return dist_1, dist_2
        return subseq1, subseq2

    @staticmethod
    def lcs_dist_list(branch, other_branches, pt_root_to_char_dict):
        seq1 = branch.get_sequence(pt_root_to_char_dict=pt_root_to_char_dict)
        other_seqs = [other_branch.get_sequence(pt_root_to_char_dict=pt_root_to_char_dict) for other_branch in other_branches]
        all_lcs = GenericBranchSeq.lcs_list(seq1, other_seqs)
        dists = []
        for lcs, branch2 in zip(all_lcs, other_branches):
            if lcs > 2:
                dists.append((lcs, GenericBranchSeq.lcs_dist(branch, branch2, pt_root_to_char_dict)))
            else:
                dists.append((lcs, (None, None)))
        
        return dists

    @staticmethod
    @jit(nopython=True)
    def sample_permutation(score_matrix):        
        # score_matrix is shaped L x A, 
        # where L is number of synapses on dendrite and A is number of axons to be sampled

        # L, A = score_matrix.shape
        L = A = score_matrix.shape[0]
        chosen_rows = np.arange(L)
        chosen_columns = np.empty(A, dtype=np.int32)
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
    
    @staticmethod
    @jit(parallel=True, fastmath=True, nopython=False)
    def sample_continuous_permutation(all_mus, all_sigmas, all_probs):
        # iterate over cell_ids (keys of dicts)
        # for each cell_id, sample an offset from the probs
        # for the chosen offset, sample a position from a truncated gaussian
        # return argsort over the sampled positions from each cell_id
        positions = []
        for axonmus, axonsigmas, probs in zip(all_mus, all_sigmas, all_probs):
            offset = np.random.choice(len(probs), p=probs)
            mu = axonmus[offset]
            sigma = axonsigmas[offset]
            # a, b = (0-mu)/sigma, (1-mu)/sigma
            # val = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=1, random_state=None)[0]

            if sigma > 0:
                # a, b = (0-mu)/sigma, (1-mu)/sigma
                a, b = 0, 1
                val = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=1, random_state=None)[0]
                val = min(1, max(0, val))
            else:
                val = min(1, max(0, mu))

            positions.append(val+offset)
        return np.argsort(positions)
    
    @staticmethod
    @jit(parallel=True, fastmath=True, nopython=False)
    def shift_synapses_continuous(all_mus, all_sigmas, all_probs, syn_positions):
        # iterate over cell_ids (keys of dicts)
        # for each cell_id, sample an offset from the probs
        # for the chosen offset, sample a position from a truncated gaussian
        # return argsort over the sampled positions from each cell_id
        positions = []
        new_syn_positions = []
        for axonmus, axonsigmas, probs in zip(all_mus, all_sigmas, all_probs):
            offset = np.random.choice(len(probs), p=probs)
            vec = syn_positions[offset+1]-syn_positions[offset]
            origin = syn_positions[offset]
            mu = axonmus[offset]
            sigma = axonsigmas[offset]
            # a, b = (0-mu)/sigma, (1-mu)/sigma
            # val = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=1, random_state=None)[0]

            if sigma > 0:
                # a, b = (0-mu)/sigma, (1-mu)/sigma
                a, b = 0, 1
                val = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=1, random_state=None)[0]
                val = min(1, max(0, val))
            else:
                val = min(1, max(0, mu))
            new_syn_positions.append(origin + val*vec)
            positions.append(val+offset)
        return np.argsort(positions), new_syn_positions


    def __init__(self, path, graph, id):

        self.cell_id = graph.graph['cell_id']
        self.root_pos = graph.graph['root_pos']
        self.path = path
        self.branch_id = id
        self.syn_id_sequence = {'collapsed': []}
        self.syn_pos_sequence = {'collapsed': []}
        self.cell_id_sequence = {'collapsed': []}
        self.syn_data_sequence = {'collapsed': []}
        self.node_dict = {node: graph.nodes[node] for node in path}
        self.inc_syn_per_node = {node: len(graph.nodes[node].get('inc_synapse_pos', [])) for node in path}
        self.out_syn_per_node = {node: len(graph.nodes[node].get('out_synapse_pos', [])) for node in path}        
        self.branch_length = np.median([graph.nodes[node].get('branch_length', 0) for node in path])
    
    def reorder_node_synapses(self):
        '''Aggregate incoming and outgoing synapses at each node and then reorder'''
        def reorder_synapses(head_pos, syn_positions):
            return np.argsort(np.linalg.norm(np.array(syn_positions).squeeze() - head_pos, axis=1))


        syn_per_node = {node: self.inc_syn_per_node[node] + self.out_syn_per_node[node] for node in self.path}
        if len(self.path) == 0 or sum(syn_per_node.values()) == 0:
            return [], []
        
        synapse_position_dict = {}
        for node in self.path:
            inc_synapses = self.node_dict[node].get('inc_synapse_pos', np.zeros((0,3)))
            out_synapses = self.node_dict[node].get('out_synapse_pos', np.zeros((0,3)))
            synapse_position_dict[node] = np.vstack((inc_synapses,out_synapses))
        
        ordered_synapse_positions = []
        in_out_pattern = []
        i = 0
        while syn_per_node[self.path[i]] == 0:
            i += 1
            if i == len(self.path):
                return [], []
        
        node = self.path[i]
        if syn_per_node[node] == 1:
            head_pos = synapse_position_dict[node][0]
        else:
            syn_positions = synapse_position_dict[node]
            dists = np.linalg.norm(syn_positions - br.root_pos, axis=1)
            head_pos = syn_positions[np.argmin(dists)]
            # reordering = reorder_synapses(head_pos, syn_positions)
            # ordered_synapse_positions.extend([syn_positions[j] for j in reordering])
            # in_out_pattern.extend([is_incoming[j] for j in reordering])
            # head_pos = syn_positions[reordering[-1]]
        
        for node in self.path[i:]:
            if syn_per_node[node] == 0:
                continue
            if syn_per_node[node] == 1:
                ordered_synapse_positions.append(synapse_position_dict[node][0])
                in_out_pattern.append(1*(self.inc_syn_per_node[node] == 1))
                head_pos = synapse_position_dict[node][0]
                continue
            
            syn_positions = synapse_position_dict[node]
            is_incoming = np.array((self.inc_syn_per_node[node])*[1] + (self.out_syn_per_node[node])*[0])
            reordering = reorder_synapses(head_pos, syn_positions)
            ordered_synapse_positions.extend([syn_positions[j] for j in reordering])
            in_out_pattern.extend([is_incoming[j] for j in reordering])
            head_pos = syn_positions[reordering[-1]]
        
        return ordered_synapse_positions, in_out_pattern

    def reorder_node_inc_synapses(self):
        def order_synapses(head_pos, syn_positions):
            return np.argsort(np.linalg.norm(np.array(syn_positions).squeeze() - head_pos, axis=1))
        
        # if the path is empty or only two synapses, maintain the original ordering
        if len(self.path) == 0 or sum(self.inc_syn_per_node.values()) <= 2:
            return
        
        # if the path is only one node, choose a head synapse and order the rest
        elif len(self.path) == 1:
            syn_positions = self.node_dict[self.path[0]].get('inc_synapse_pos', [])
            if len(syn_positions) > 0:
                syn_positions = np.array(syn_positions)
                # compute distances to root_pos and set head_pos to the synapse closest to root_pos
                dists = np.linalg.norm(syn_positions - self.root_pos, axis=1)
                head_pos = syn_positions[np.argmin(dists)]
                self.node_dict[self.path[0]]['inc_synapse_pos'] = [syn_positions[i] for i in order_synapses(head_pos, syn_positions)]
            return
        
        # if the path is more than one node, choose a head synapse for the first node and order the rest
        else:
            i = 0
            while self.inc_syn_per_node[self.path[i]]==0:
                i += 1
                if i == len(self.path):
                    return

            if self.inc_syn_per_node[self.path[i]] == 1:
                head_pos = self.node_dict[self.path[i]]['inc_synapse_pos'][0]
            else: # if there are no out synapses, this node must have (multiple) incoming synapses
                syn_positions = self.node_dict[self.path[i]]['inc_synapse_pos']
                # compute distances to root_pos and set head_pos to the synapse closest to root_pos
                dists = np.linalg.norm(syn_positions - self.root_pos, axis=1)
                head_pos = syn_positions[np.argmin(dists)]
            
            # reorder synapses in each node and update the head_pos for each subsequent node
            for node in self.path[i:]:
                if self.inc_syn_per_node[node] <= 1:
                    continue
                syn_positions = self.node_dict[node]['inc_synapse_pos']
                self.node_dict[node]['inc_synapse_pos'] = [syn_positions[k] for k in order_synapses(head_pos, syn_positions)]
                head_pos = self.node_dict[node]['inc_synapse_pos'][-1]
        return

    def collapse_path(self, filter_dict):

        # if some nodes have multiple synapses, reorder each node's synapses
        if sum(self.inc_syn_per_node.values()) != len(self.path):
            self.reorder_node_inc_synapses()

        last_cid = None
        for node in self.path:
            num_syn = self.inc_syn_per_node[node]
            for syn in range(num_syn):
                if all([self.node_dict[node].get(key, num_syn*[None])[syn] in values for key, values in filter_dict.items()]):
                    if last_cid is None or self.node_dict[node]['pre_cell_id'][syn] != last_cid:
                        self.syn_pos_sequence['collapsed'].append(self.node_dict[node]['inc_synapse_pos'][syn])
                        self.cell_id_sequence['collapsed'].append(self.node_dict[node]['pre_cell_id'][syn])
                        self.syn_data_sequence['collapsed'].append({k: v[syn] for k,v in self.node_dict[node].items() if k not in ['post_cell_id', 'out_synapse_pos']})
                        last_cid = self.node_dict[node]['pre_cell_id'][syn]
        self.collapsed_cell_id_set = set(self.cell_id_sequence['collapsed'])
        return

    def length(self, collapsed=True):
        if collapsed:
            return len(self.syn_pos_sequence['collapsed'])
        else:
            return len(self.path)
    
    def subsequence(self, indices):
        new_branch = deepcopy(self)
        new_branch.syn_pos_sequence['collapsed'] = [self.syn_pos_sequence['collapsed'][i] for i in indices]
        new_branch.cell_id_sequence['collapsed'] = [self.cell_id_sequence['collapsed'][i] for i in indices]
        new_branch.syn_data_sequence['collapsed'] = [self.syn_data_sequence['collapsed'][i] for i in indices]
        return new_branch
    
    def get_sequence(self, pt_root_to_char_dict=None):
        return GenericBranchSeq.sequence_to_string(self.cell_id_sequence['collapsed'], pt_root_to_char_dict)

    def distance(self, collapsed=True):
        if collapsed:
            pos = np.array(self.syn_pos_sequence['collapsed'], dtype=np.float32)
            if len(pos) < 2:
                return 0
            return np.linalg.norm(pos[1:] - pos[:-1], axis=1).sum()
        else:
            pos = []
            for node in self.path:
                if self.inc_syn_per_node[node]>0:
                    pos.append(self.node_dict[node]['inc_synapse_pos'][0])
                else:
                    pos.append(self.node_dict[node]['out_synapse_pos'][0])
            pos = np.array(pos)
            if len(pos) < 2:
                return 0
            return np.linalg.norm(pos[1:] - pos[:-1], axis=1).sum()
        
    def get_random_shuffle(self, collapsed=True):
        if not collapsed:
            return ValueError("Can only shuffle collapsed sequences")
        if len(self.cell_id_sequence['collapsed']) < 2 or len(self.collapsed_cell_id_set) < 2:
            random_branch = DummyBranchSeq(cell_id=self.cell_id, branch_id=self.branch_id, branch_length=self.branch_length,
                                           syn_pos_sequence=self.syn_pos_sequence, cell_id_sequence=self.cell_id_sequence, 
                                           syn_data_sequence=self.syn_data_sequence)
            return random_branch
        
        permutation = np.random.permutation(len(self.cell_id_sequence['collapsed']))
        cell_id_sequence = {'collapsed': [self.cell_id_sequence['collapsed'][i] for i in permutation]}
        syn_data_sequence = {'collapsed': [self.syn_data_sequence['collapsed'][i] for i in permutation]}
        random_branch = DummyBranchSeq(cell_id=self.cell_id, branch_id=self.branch_id, branch_length=self.branch_length,
                                       syn_pos_sequence=self.syn_pos_sequence,
                                       cell_id_sequence=cell_id_sequence, syn_data_sequence=syn_data_sequence)
        return random_branch
    
    def get_type_shuffle(self, data_key, collapsed=True):
        if not collapsed:
            return ValueError("Can only shuffle collapsed sequences")
        if len(self.cell_id_sequence['collapsed']) < 2 or len(self.collapsed_cell_id_set) < 2:
            random_branch = DummyBranchSeq(cell_id=self.cell_id, branch_id=self.branch_id, branch_length=self.branch_length,
                                           syn_id_sequence=self.syn_id_sequence, syn_pos_sequence=self.syn_pos_sequence,
                                           cell_id_sequence=self.cell_id_sequence, syn_data_sequence=self.syn_data_sequence)
            return random_branch
        
        synapses_by_type = defaultdict(list)
        for cell_id, syn_data in zip(self.cell_id_sequence['collapsed'], self.syn_data_sequence['collapsed']):
            cell_type = syn_data[data_key]
            synapses_by_type[cell_type].append((cell_id, syn_data))

        for cell_type in synapses_by_type:
            np.random.shuffle(synapses_by_type[cell_type])
        
        new_cell_id_sequence = {'collapsed': []}
        new_syn_data_sequence = {'collapsed': []}
        for old_syn_data in self.syn_data_sequence['collapsed']:
            cell_type = old_syn_data[data_key]
            cell_id, new_syn_data = synapses_by_type[cell_type].pop()
            new_cell_id_sequence['collapsed'].append(cell_id)
            new_syn_data_sequence['collapsed'].append(new_syn_data)
        
        random_branch = DummyBranchSeq(cell_id=self.cell_id, branch_id=self.branch_id, branch_length=self.branch_length,
                                       syn_pos_sequence=self.syn_pos_sequence,
                                       cell_id_sequence=new_cell_id_sequence, syn_data_sequence=new_syn_data_sequence)
        return random_branch

    def get_axon_shuffle(self, score_mat, collapsed=True):
        if not collapsed:
            return ValueError("Can only shuffle collapsed sequences")
        if len(score_mat) < 2 or len(self.collapsed_cell_id_set) < 2:
            random_branch = DummyBranchSeq(cell_id=self.cell_id, branch_id=self.branch_id, branch_length=self.branch_length,
                                           syn_pos_sequence=self.syn_pos_sequence,
                                           cell_id_sequence=self.cell_id_sequence, syn_data_sequence=self.syn_data_sequence)
            return random_branch
        
        permutation = GenericBranchSeq.sample_permutation(np.array(score_mat).copy())
        cell_id_sequence = {'collapsed': [self.cell_id_sequence['collapsed'][i] for i in permutation]}
        syn_data_sequence = {'collapsed': [self.syn_data_sequence['collapsed'][i] for i in permutation]}
        random_branch = DummyBranchSeq(cell_id=self.cell_id, branch_id=self.branch_id, branch_length=self.branch_length,
                                       syn_pos_sequence=self.syn_pos_sequence,
                                       cell_id_sequence=cell_id_sequence, syn_data_sequence=syn_data_sequence)
        return random_branch

    def get_continuous_shuffle(self, dicts, collapsed=True, edit_position=False):
        if not collapsed:
            return ValueError("Can only shuffle collapsed sequences")
        if dicts is None or len(self.collapsed_cell_id_set) < 2:
            random_branch = DummyBranchSeq(cell_id=self.cell_id, branch_id=self.branch_id, branch_length=self.branch_length,
                                           syn_pos_sequence=self.syn_pos_sequence,
                                           cell_id_sequence=self.cell_id_sequence, syn_data_sequence=self.syn_data_sequence)
            return random_branch

        def clean_probs(p):
            # set nans to zero
            p[np.isnan(p)] = 0
            # set negative values to zero
            p[p < 0] = 0
            if p.sum() == 0:
                return np.ones_like(p) / len(p)
            return p / p.sum()
        
        mus = [axondicts['mus'] for axondicts in dicts]
        sigmas = [axondicts['sigmas'] for axondicts in dicts]
        probs = [clean_probs(axondicts['probs']) for axondicts in dicts]
        if edit_position:
            perm, new_pos = GenericBranchSeq.shift_synapses_continuous(mus, sigmas, probs, self.syn_pos_sequence['collapsed'])
            cell_id_sequence = {'collapsed': [self.cell_id_sequence['collapsed'][i] for i in perm]}
            syn_data_sequence = {'collapsed': [self.syn_data_sequence['collapsed'][i] for i in perm]}
            random_branch = DummyBranchSeq(cell_id=self.cell_id, branch_id=self.branch_id, branch_length=self.branch_length,
                                             syn_pos_sequence={'collapsed': new_pos},
                                             cell_id_sequence=cell_id_sequence, syn_data_sequence=syn_data_sequence)
            return random_branch
        
        perm = GenericBranchSeq.sample_continuous_permutation(mus, sigmas, probs)
        cell_id_sequence = {'collapsed': [self.cell_id_sequence['collapsed'][i] for i in perm]}
        syn_data_sequence = {'collapsed': [self.syn_data_sequence['collapsed'][i] for i in perm]}
        random_branch = DummyBranchSeq(cell_id=self.cell_id, branch_id=self.branch_id, branch_length=self.branch_length,
                                       syn_pos_sequence=self.syn_pos_sequence,
                                       cell_id_sequence=cell_id_sequence, syn_data_sequence=syn_data_sequence)
        return random_branch
    
class DummyBranchSeq(GenericBranchSeq):
    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    