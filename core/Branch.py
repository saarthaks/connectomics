import numpy as np
from numba import jit
import networkx as nx
from copy import deepcopy
from collections import defaultdict
import pylcs as LCS
from tqdm import tqdm

# from numba_stats import truncnorm

# from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
# import warnings

# warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
# warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)

class BranchSeq:

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
        all_lcs = BranchSeq.lcs_list(seq1, other_seqs)
        dists = []
        for lcs, branch2 in zip(all_lcs, other_branches):
            if lcs > 2:
                dists.append((lcs, BranchSeq.lcs_dist(branch, branch2, pt_root_to_char_dict)))
            else:
                dists.append((lcs, (None, None)))
        
        return dists
    
    @staticmethod
    def string_list(seq1, other_seqs):
        return LCS.lcs_string_of_list(seq1, other_seqs)

    @staticmethod
    def string_dist(branch1, branch2, pt_root_to_char_dict):
        seq1 = branch1.get_sequence(pt_root_to_char_dict=pt_root_to_char_dict)
        seq2 = branch2.get_sequence(pt_root_to_char_dict=pt_root_to_char_dict)

        seq_1_idx = list(filter(lambda i: i != -1, LCS.lcs_string_idx(seq2, seq1)))
        subseq1 = branch1.subsequence(seq_1_idx)
        # min_1, max_1 = min(seq_1_idx), max(seq_1_idx)
        # dist_1 = np.linalg.norm(branch1.syn_pos_sequence['collapsed'][min_1] - branch1.syn_pos_sequence['collapsed'][max_1])
        
        seq_2_idx = list(filter(lambda i: i != -1, LCS.lcs_string_idx(seq1, seq2)))
        subseq2 = branch2.subsequence(seq_2_idx)
        # min_2, max_2 = min(seq_2_idx), max(seq_2_idx)
        # dist_2 = np.linalg.norm(branch2.syn_pos_sequence['collapsed'][min_2] - branch2.syn_pos_sequence['collapsed'][max_2])

        # return dist_1, dist_2
        return subseq1, subseq2

    @staticmethod
    def string_dist_list(branch, other_branches, pt_root_to_char_dict):
        seq1 = branch.get_sequence(pt_root_to_char_dict=pt_root_to_char_dict)
        other_seqs = [other_branch.get_sequence(pt_root_to_char_dict=pt_root_to_char_dict) for other_branch in other_branches]
        all_lcs = BranchSeq.string_list(seq1, other_seqs)
        dists = []
        for lcs, branch2 in zip(all_lcs, other_branches):
            if lcs > 2:
                dists.append((lcs, BranchSeq.string_dist(branch, branch2, pt_root_to_char_dict)))
            else:
                dists.append((lcs, (None, None)))
        
        return dists
    
    @staticmethod
    def cluster_dist(branch1, branch2):
        set1 = set(branch1.cell_id_sequence['collapsed'])
        set2 = set(branch2.cell_id_sequence['collapsed'])
        over = set1.intersection(set2)

        set1_idx = np.sort(np.argwhere(np.isin(branch1.cell_id_sequence['collapsed'], list(over))).flatten())
        subseq1 = branch1.subsequence(set1_idx)
        # min_1, max_1 = min(set1_idx), max(set1_idx)
        # dist_1 = np.linalg.norm(branch1.syn_pos_sequence['collapsed'][min_1] - branch1.syn_pos_sequence['collapsed'][max_1])

        set2_idx = np.sort(np.argwhere(np.isin(branch2.cell_id_sequence['collapsed'], list(over))).flatten())
        subseq2 = branch2.subsequence(set2_idx)
        # min_2, max_2 = min(set2_idx), max(set2_idx)
        # dist_2 = np.linalg.norm(branch2.syn_pos_sequence['collapsed'][min_2] - branch2.syn_pos_sequence['collapsed'][max_2])

        # return dist_1, dist_2
        return subseq1, subseq2

    @staticmethod
    def cluster_dist_list(branch, other_branches, pt_root_to_char_dict):
        set1 = set(branch.cell_id_sequence['collapsed'])
        other_sets = [set(other_branch.cell_id_sequence['collapsed']) for other_branch in other_branches]
        all_cluster_sizes = [len(set1.intersection(other_set)) for other_set in other_sets]
        dists = []
        for cluster_size, branch2 in zip(all_cluster_sizes, other_branches):
            if cluster_size > 1:
                dists.append((cluster_size, BranchSeq.cluster_dist(branch, branch2)))
            else:
                dists.append((cluster_size, (np.inf, np.inf)))
        
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


    def __init__(self, path, graph, id, valid_types={'23P', '4P', '5P-IT', '5P-ET', '5P-NP', '6P-IT', '6P-CT'}):

        self.cell_id = graph.graph['cell_id']
        self.branch_id = id
        self.syn_id_sequence = {'raw': [], 'collapsed': []}
        self.syn_pos_sequence = {'raw': [], 'collapsed': []}
        self.cell_id_sequence = {'raw': [], 'collapsed': []}
        self.cell_type_sequence = {'raw': [], 'collapsed': []}

        if len(path) > 0:
            try:
                if graph.nodes[path[0]]['cell_type']:
                    cell_type_key = 'cell_type'
            except KeyError:
                cell_type_key = 'cell_type_pre'
        else:
            cell_type_key = 'cell_type'
        
        for node in path:
            if graph.nodes[node].get(cell_type_key, 'Unknown') not in valid_types:
                continue
            self.syn_id_sequence['raw'].append(graph.nodes[node].get('syn_id', node))
            self.syn_pos_sequence['raw'].append(graph.nodes[node]['pos'])
            self.cell_id_sequence['raw'].append(graph.nodes[node]['pre_cell_id'])
            self.cell_type_sequence['raw'].append(graph.nodes[node][cell_type_key])
        
        collapsed_path = BranchSeq.collapse_path(path, graph)
        for node in collapsed_path:
            if graph.nodes[node].get(cell_type_key, 'Unknown') not in valid_types:
                continue
            self.syn_id_sequence['collapsed'].append(graph.nodes[node].get('syn_id', node))
            self.syn_pos_sequence['collapsed'].append(graph.nodes[node]['pos'])
            self.cell_id_sequence['collapsed'].append(graph.nodes[node]['pre_cell_id'])
            self.cell_type_sequence['collapsed'].append(graph.nodes[node][cell_type_key])
        
        self.branch_length = np.median([graph.nodes[node].get('branch_length', 0) for node in path])
        self.cell_id_set = set(self.cell_id_sequence['collapsed'])
        self.apical = False

    
    def length(self, collapsed=True):
        if collapsed:
            return len(self.syn_id_sequence['collapsed'])
        else:
            return len(self.syn_id_sequence['raw'])
    
    def subsequence(self, indices, collapsed=True):
        if collapsed:
            new_branch = deepcopy(self)
            new_branch.syn_id_sequence['collapsed'] = [self.syn_id_sequence['collapsed'][i] for i in indices]
            new_branch.syn_pos_sequence['collapsed'] = [self.syn_pos_sequence['collapsed'][i] for i in indices]
            new_branch.cell_id_sequence['collapsed'] = [self.cell_id_sequence['collapsed'][i] for i in indices]
            new_branch.cell_type_sequence['collapsed'] = [self.cell_type_sequence['collapsed'][i] for i in indices]
            return new_branch
        else:
            new_branch = deepcopy(self)
            new_branch.syn_id_sequence['raw'] = [self.syn_id_sequence['raw'][i] for i in indices]
            new_branch.syn_pos_sequence['raw'] = [self.syn_pos_sequence['raw'][i] for i in indices]
            new_branch.cell_id_sequence['raw'] = [self.cell_id_sequence['raw'][i] for i in indices]
            new_branch.cell_type_sequence['raw'] = [self.cell_type_sequence['raw'][i] for i in indices]
            return new_branch
    
    def get_sequence(self, collapsed=True, pt_root_to_char_dict=None):
        if collapsed:
            return BranchSeq.sequence_to_string(self.cell_id_sequence['collapsed'], pt_root_to_char_dict)
        else:
            return BranchSeq.sequence_to_string(self.cell_id_sequence['raw'], pt_root_to_char_dict)

    def distance(self, collapsed=True):
        if collapsed:
            pos = np.array(self.syn_pos_sequence['collapsed'], dtype=np.float32)
            if len(pos) < 2:
                return 0
            return np.linalg.norm(pos[1:] - pos[:-1], axis=1).sum()
        else:
            pos = np.array(self.syn_pos_sequence['raw'])
            if len(pos) < 2:
                return 0
            return np.linalg.norm(pos[1:] - pos[:-1], axis=1).sum()
        
    def get_random_shuffle(self, collapsed=True):
        if not collapsed:
            return ValueError("Can only shuffle collapsed sequences")
        random_branch = deepcopy(self)
        if len(self.cell_id_sequence['collapsed']) < 2:
            return random_branch
        
        permutation = np.random.permutation(len(self.syn_id_sequence['collapsed']))
        random_branch.syn_id_sequence['collapsed'] = [self.syn_id_sequence['collapsed'][i] for i in permutation]
        random_branch.cell_id_sequence['collapsed'] = [self.cell_id_sequence['collapsed'][i] for i in permutation]
        random_branch.cell_type_sequence['collapsed'] = [self.cell_type_sequence['collapsed'][i] for i in permutation]
        return random_branch
    
    def get_type_shuffle(self, collapsed=True):
        if not collapsed:
            return ValueError("Can only shuffle collapsed sequences")
        random_branch = deepcopy(self)
        if len(self.cell_id_sequence['collapsed']) < 2:
            return random_branch
        
        synapses_by_type = {}
        for cell_id, syn_id, cell_type in zip(self.cell_id_sequence['collapsed'], self.syn_id_sequence['collapsed'], self.cell_type_sequence['collapsed']):
            if cell_type not in synapses_by_type:
                synapses_by_type[cell_type] = []
            synapses_by_type[cell_type].append((cell_id, syn_id))

        for cell_type in synapses_by_type:
            np.random.shuffle(synapses_by_type[cell_type])
        
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
        random_branch = deepcopy(self)
        if len(score_mat) < 2:
            return random_branch
        
        perm = BranchSeq.sample_permutation(np.array(score_mat).copy())
        random_branch.syn_id_sequence['collapsed'] = [self.syn_id_sequence['collapsed'][i] for i in perm]
        random_branch.cell_id_sequence['collapsed'] = [self.cell_id_sequence['collapsed'][i] for i in perm]
        random_branch.cell_type_sequence['collapsed'] = [self.cell_type_sequence['collapsed'][i] for i in perm]
        return random_branch

    def get_continuous_shuffle(self, dicts, collapsed=True, edit_position=False):
        if not collapsed:
            return ValueError("Can only shuffle collapsed sequences")
        random_branch = deepcopy(self)
        if dicts is None:
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
            perm, new_pos = BranchSeq.shift_synapses_continuous(mus, sigmas, probs, self.syn_pos_sequence['collapsed'])
            random_branch.syn_id_sequence['collapsed'] = [self.syn_id_sequence['collapsed'][i] for i in perm]
            random_branch.cell_id_sequence['collapsed'] = [self.cell_id_sequence['collapsed'][i] for i in perm]
            random_branch.cell_type_sequence['collapsed'] = [self.cell_type_sequence['collapsed'][i] for i in perm]
            random_branch.syn_pos_sequence['collapsed'] = new_pos
            return random_branch
        
        perm = BranchSeq.sample_continuous_permutation(mus, sigmas, probs)
        random_branch.syn_id_sequence['collapsed'] = [self.syn_id_sequence['collapsed'][i] for i in perm]
        random_branch.cell_id_sequence['collapsed'] = [self.cell_id_sequence['collapsed'][i] for i in perm]
        random_branch.cell_type_sequence['collapsed'] = [self.cell_type_sequence['collapsed'][i] for i in perm]
        return random_branch