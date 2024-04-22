import argparse
import pickle
import networkx as nx
import pandas as pd
from sklearn.neighbors import KDTree
from tqdm import tqdm

import pickle
from sklearn.mixture import GaussianMixture
import numpy as np
from scipy.stats import multivariate_normal

class ReducedGaussianMixture:
    def __init__(self, means, covariances, weights):
        self.means_ = means
        self.covariances_ = covariances
        self.weights_ = weights
        self.n_components = len(weights)
        self.n_features = 3
        self.covariance_type = 'full'
    
    def score_samples(self, X):
        return np.log(np.array([self.weights_[i] * multivariate_normal.pdf(X, mean=self.means_[i], cov=self.covariances_[i]) for i in range(len(self.weights_))]).sum(axis=0))

def main(gmm_dict_path, branches_path, score_dict_path):
    with open(gmm_dict_path, 'rb') as f:
        gmms_dict = pickle.load(f)

    with open(branches_path, 'rb') as f:
        all_branches = pickle.load(f)

    score_dict = {}
    for key in tqdm(all_branches.keys()):
        if key in score_dict:
            continue
        score_dict[key] = []
        for branch in all_branches[key]:
            cell_id_sequence = branch.cell_id_sequence['collapsed']
            branch_scores = []
            for cell_id in cell_id_sequence:
                gmm = gmms_dict[cell_id]
                scores = np.exp(gmm.score_samples(branch.syn_pos_sequence['collapsed']))
                branch_scores.append(scores)
            score_dict[key].append(branch_scores)

    with open(score_dict_path, 'wb') as f:
        pickle.dump(score_dict, f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gmm_dict_path', type=str, required=True) 
    parser.add_argument('--branches_path', type=str, required=True)
    parser.add_argument('--score_dict_path', type=str, required=True)
    args = parser.parse_args()

    main(args.gmm_dict_path, args.branches_path, args.score_dict_path)