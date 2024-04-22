import pickle
import networkx as nx
import pandas as pd
from sklearn.neighbors import KDTree

with open('data/axons/all_axons.pkl', 'rb') as f:
    all_axons = pickle.load(f)

# create kd tree with synapses from all axon graphs
all_synapse_dfs = []
for axon in all_axons:
    all_synapse_dfs.append(pd.DataFrame.from_dict(graph.nodes, orient='index'))

axn_syn_df = pd.concat(all_synapse_dfs)
kd_tree = KDTree(axn_syn_df.values)

with open('data/gmms_dict.pkl', 'rb') as f:
    gmms_dict = pickle.load(f)

with open('data/all_sequence_data.pkl', 'rb') as f:
    sequence_positions = pickle.load(f)

score_dict = {}
for key in tqdm(sequence_positions):
    score_dict[key] = []
    for branch in sequence_positions[key]:
        cell_id_sequence = [char_to_pt_root_id[char] for char in branch[0]]
        pre_cell_ids = []
        for syn_ids in kd_tree.query_radius(branch[2], r=500):
            pre_cell_ids.extend(xn_syn_df.iloc[syn_ids]['pre_cell_id'].values)
        pre_cell_ids = list(set(pre_cell_ids))

        branch_scores = []
        for cell_id in cell_id_sequence:
            gmm = gmms_dict[cell_id]
            scores = gmm.score_samples(branch[1])
            branch_scores.append(scores)
        for cell_id in pre_cell_ids:
            gmm = gmms_dict[cell_id]
            scores = gmm.score_scamples(branch[1])
            branch_scores.append(scores)
        score_dict[key].append(branch_scores)

with open('data/adp_score_dict.pkl', 'wb') as f:
    pickle.dump(score_dict, f)