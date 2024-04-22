import pickle
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
from core.Branch import BranchSeq

def main():

    with open('/drive_sdc/ssarup/connectomics_data/data/all_sequence_data_dict.pkl', 'rb') as f:
        sequence_data = pickle.load(f)


    neuron_ids = list(sequence_data.keys())

    all_branches = {}
    for neuron_id in tqdm(neuron_ids):
        G = nx.Graph(cell_id=neuron_id)

        all_branches[neuron_id] = []
        i = 0
        for cell_seq, syn_seq, pos_seq, type_seq in sequence_data[neuron_id]:
            branch = BranchSeq([], G, hash((neuron_id, i)))
            branch.syn_id_sequence['raw'] = syn_seq
            branch.cell_id_sequence['raw'] = cell_seq
            branch.cell_type_sequence['raw'] = type_seq
            branch.syn_pos_sequence['raw'] = pos_seq
            branch.syn_id_sequence['collapsed'] = syn_seq
            branch.cell_id_sequence['collapsed'] = cell_seq
            branch.cell_type_sequence['collapsed'] = type_seq
            branch.syn_pos_sequence['collapsed'] = pos_seq
            all_branches[neuron_id].append(branch)
            i += 1
    
    with open('/drive_sdc/ssarup/microns_data/branches/all_branches_og.pkl', 'wb') as f:
        pickle.dump(all_branches, f)
    
    with open('/drive_sdc/ssarup/connectomics_data/data/pair_list.pkl', 'rb') as f:
        pair_list = pickle.load(f)
    
    neuron_sim_dict = {}
    for a,b in tqdm(pair_list):
        if a in all_branches and b in all_branches:
            if a in neuron_sim_dict:
                neuron_sim_dict[a].append(b)
            elif b in neuron_sim_dict:
                neuron_sim_dict[b].append(a)
            else:
                neuron_sim_dict[a] = [b]
        else:
            print(f'Pair {a} {b} not in all_branches!')

    
    branch_sim_dict = defaultdict(list)
    for neuron_id in tqdm(neuron_sim_dict):
        branches = all_branches[neuron_id]
        branch_inp_cells_1 = [set(branch.cell_id_sequence['raw']) for branch in branches]

        for other_neuron_id in neuron_sim_dict[neuron_id]:
            other_branches = all_branches[other_neuron_id]
            branch_inp_cells_2 = [set(branch.cell_id_sequence['raw']) for branch in other_branches]

            for branch_id_1, inp_cells_1 in enumerate(branch_inp_cells_1):
                for branch_id_2, inp_cells_2 in enumerate(branch_inp_cells_2):
                    if len(inp_cells_1.intersection(inp_cells_2)) > 1:
                        branch_sim_dict[neuron_id].append((other_neuron_id, branch_id_1, branch_id_2))
    
    with open('/drive_sdc/ssarup/microns_data/similarity_dict_og.pkl', 'wb') as f:
        pickle.dump(branch_sim_dict, f)

if __name__ == '__main__':
    main()
            

