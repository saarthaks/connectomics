import argparse
import pickle
import networkx as nx
import pandas as pd
from sklearn.neighbors import KDTree
from tqdm import tqdm

import pickle
from sklearn.mixture import GaussianMixture
import numpy as np
from scipy.stats import multivariate_normal, norm

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

    def predict(self, X):
        return np.argmax(np.array([self.weights_[i] * multivariate_normal.pdf(X, mean=self.means_[i], cov=self.covariances_[i]) for i in range(len(self.weights_))]), axis=0)
    
    def predict_proba(self, X):
        return np.array([self.weights_[i] * multivariate_normal.pdf(X, mean=self.means_[i], cov=self.covariances_[i]) for i in range(len(self.weights_))]).T

def probability_dict(gmm, positions, component):
    all_mus = []
    all_sigmas = []
    line_probs = []
    seg_probs = []
    all_probs = []
    offset = 0
    for origin, terminal in zip(positions[:-1], positions[1:]):
        vec = terminal-origin
        # create a permutation matrix ordering vec from largest to smallest
        perm = np.argsort(np.abs(vec))[::-1]
        P = np.eye(3)[perm]
        pvec = vec[perm]
        porigin = origin[perm]
        Mu = gmm.means_[component]
        Sigma = gmm.covariances_[component]
        # rotate the data so that the vector is along the x-axis
        Mu = P @ Mu
        Sigma = P @ Sigma @ P.T


        N = np.zeros((2, 4)); N[0,-1] = 1
        N[0,:-1] = porigin
        N[1,:-1] = pvec
        _, _, V = np.linalg.svd(N)
        null_space = V[-2:,:]

        A = np.zeros((3,3))
        A[0,0] = 1
        A[1] = null_space[0,:-1]+null_space[1,:-1]
        A[2] = null_space[0,:-1]-null_space[1,:-1]
        c = np.array([-null_space[0,-1]-null_space[1,-1], -null_space[0,-1]+null_space[1,-1]])

        new_mu = A @ Mu
        new_sigma = A @ Sigma @ A.T
        # new_mu = A @ gmm.means_[component]
        # new_sigma = A @ gmm.covariances_[component] @ A.T

        new_new_mu = new_mu[0] + new_sigma[0,1:3] @ np.linalg.inv(new_sigma[1:3,1:3]) @ (c - new_mu[1:3])
        new_new_sigma = new_sigma[0,0] - new_sigma[0,1:3] @ np.linalg.inv(new_sigma[1:3,1:3]) @ new_sigma[1:3,0]

        mu_t = (new_new_mu - porigin[0])/pvec[0]
        sigma_t = np.sqrt(np.abs(new_new_sigma) / pvec[0]**2)
        all_mus.append(mu_t)
        all_sigmas.append(sigma_t)

        # total_prob = np.diff(norm.cdf([0, 1], mu_t, sigma_t))
        # total_prob = np.mean(multivariate_normal.pdf([origin, terminal], mean=gmm.means_[component], cov=gmm.covariances_[component]))*np.linalg.norm(vec)
        total_prob = multivariate_normal.pdf(origin, mean=gmm.means_[component], cov=gmm.covariances_[component])
        all_probs.append(total_prob)

        # nvec = pvec / np.linalg.norm(pvec)
        # w = np.array([0, nvec[2], -nvec[1]])
        # V = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
        # R = np.eye(3) + V + V @ V / (1 + nvec[0])
        # amu = R @ (Mu - origin)
        # asigma = R @ Sigma @ R.T
        # aP = np.linalg.inv(asigma)
        # a = aP[0,0]/2
        # b = (aP@amu)[0]
        # c = amu @ aP @ amu / 2
        # line_prob = np.log(np.sqrt(np.pi/a)) + (b**2/(4*a) - c) - np.log((2*np.pi)**(3/2)) - np.log(np.sqrt(np.linalg.det(asigma)))
        # seg_prob = np.abs(np.diff(norm.cdf([0, 1], loc=mu_t, scale=sigma_t)))[0]
        # if seg_prob == 0:
        #     seg_prob = -100
        # else:
        #     seg_prob = np.log(seg_prob)

        # line_probs.append(line_prob)
        # seg_probs.append(seg_prob)
        offset += 1

    # line_probs = np.array(line_probs).squeeze()
    # if line_probs.sum() == 0:
    #     line_probs = -np.log(np.len(line_probs)) * np.ones_like(line_probs)
    # seg_probs = np.array(seg_probs).squeeze()
    # all_probs = line_probs + seg_probs
    # all_probs = np.exp(all_probs - np.max(all_probs))
    all_probs = np.array(all_probs).squeeze()
    all_probs = all_probs / all_probs.sum()
    return {'mus': all_mus, 'sigmas': all_sigmas, 'probs': all_probs}

def main(gmm_dict_path, branches_path, prob_dict_path):
    with open(gmm_dict_path, 'rb') as f:
        gmms_dict = pickle.load(f)

    with open(branches_path, 'rb') as f:
        all_branches = pickle.load(f)

    prob_dict = {}
    for key in tqdm(all_branches.keys()):
        prob_dict[key] = []
        for branch in all_branches[key]:
            cell_id_sequence = branch.cell_id_sequence['collapsed']
            big_syn_pos_sequence = [pos.reshape(1,3) for pos in branch.syn_pos_sequence['collapsed']]
            syn_pos_sequence = [pos.squeeze() for pos in branch.syn_pos_sequence['collapsed']]
            if len(cell_id_sequence) >= 3:
                mixture_components = {cell_id: gmms_dict[cell_id].predict(pos).squeeze() for pos, cell_id in zip(big_syn_pos_sequence, cell_id_sequence)}
                dicts = [probability_dict(gmms_dict[cell_id], syn_pos_sequence, mixture_components[cell_id]) for cell_id in cell_id_sequence]
                prob_dict[key].append(dicts)
            else:
                prob_dict[key].append(None)
    
    with open(prob_dict_path, 'wb') as f:
        pickle.dump(prob_dict, f)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gmm_dict_path', type=str, required=True) 
    parser.add_argument('--branches_path', type=str, required=True)
    parser.add_argument('--prob_dict_path', type=str, required=True)
    args = parser.parse_args()

    main(args.gmm_dict_path, args.branches_path, args.prob_dict_path)