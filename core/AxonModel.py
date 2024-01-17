import numpy as np
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from .Skeleton import Skeleton

class AxonModel(Skeleton):

    @staticmethod
    def create_axon_score_dict(axon_models, branches, collapsed=True):
        score_dict = {}
        for branch in tqdm(branches):
            cell_seq = branch.cell_id_sequence['collapsed'] if collapsed else branch.cell_id_sequence['raw']
            pos_seq = branch.syn_pos_sequence['collapsed'] if collapsed else branch.syn_pos_sequence['raw']
            branch_scores = []
            for cell_id in cell_seq:
                scores = axon_models[cell_id].gmm.score_samples(pos_seq)
                branch_scores.append(scores)
            score_dict[branch.branch_id] = branch_scores
        
        return score_dict

    def __init__(self, cell_info, syn_group, syn_k=8, soma_k=8, twig_length=4, single_syn_std=5):
        self.single_syn_std = single_syn_std
        super().__init__(cell_info, syn_group, syn_k, soma_k)
        self.smooth(twig_length, prune_unknown=False)
        self.fit_gmm()

    def fit_gmm(self, min_path_length=4):
        paths = self.get_paths(smoothed=True, exc=False, duplicate_tail=True)
        all_positions = []
        all_means = []
        all_precisions = []
        for path in paths:
            path_positions = np.array([self.smooth_mst.nodes[node]['pos'] for node in path])
            mean = np.mean(path_positions, axis=0)
            if len(path_positions) >= min_path_length:
                precision = np.linalg.inv(np.cov(path_positions.T))
            elif len(path_positions) == 1:
                precision = np.diag(1/(np.array(3*[self.single_syn_std]))**2)
            else:
                variance = np.var(path_positions, axis=0)
                variance[variance == 0] = self.single_syn_std**2
                precision = np.diag(1/variance)
            
            precision[np.isinf(precision)] = 1/(self.single_syn_std**2)
            all_positions.append(path_positions)
            all_means.append(mean)
            all_precisions.append(precision)

        all_positions = np.concatenate(all_positions, axis=0)
        gmm = GaussianMixture(n_components=len(all_means), covariance_type='full', 
                means_init=np.array(all_means),
                precisions_init=np.array(all_precisions))
        if all_positions.shape[0] == 1:
            all_positions = np.concatenate([all_positions, all_positions], axis=0)
        gmm.fit(all_positions)
        self.gmm = gmm