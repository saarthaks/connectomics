import pandas as pd
import os

from core.flywire_utils import *
import collapse_filter_branches
import shuffle_lcs_experiment
import make_lcs_histograms
import make_klcs_histograms
import make_sequencedist_distributions

if __name__ == '__main__':
    base = '/drive_sdc/ssarup/flywire_data'
    figures_base = '/home/ssarup/flywire_figures'

    neuron_annotation = pd.read_csv('./neuron_annotation.tsv', sep='\t')

    ## Step 1: Choose a subset of neurons to analyze
    region = 'olfactory'
    os.makedirs(os.path.join(base, region), exist_ok=True)

    cell_ids = neuron_annotation[neuron_annotation['cell_class'] == 'olfactory']['root_id'].values
    branches = load_branches_dict(cell_ids)
    pre_ids = get_pre_ids(branches)
    pre_ids_df = neuron_annotation[neuron_annotation['root_id'].isin(pre_ids)]
    pre_ids_df.to_csv(os.path.join(base, region, 'pre_ids.csv'), index=False)

    # Step 2: Collapse branches based on filter_dict
    experiment_name = 'dopamine'
    filter_dict = {
        'top_nts': ['dopamine'],
    }
    min_intersection = 3
    output_path = os.path.join(base, region, experiment_name)
    os.makedirs(output_path, exist_ok=True)
    collapse_filter_branches.main(branches, output_path, filter_dict,
                                  min_intersection=min_intersection,
                                  verbose=True)
    
    # Step 3: Run the LCS experiment (for now, just the original and the random shuffles)
    os.makedirs(os.path.join(output_path, 'original'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'random'), exist_ok=True)
    collapsed_path = os.path.join(output_path, 'collapsed_branches.pkl')
    similarity_path = os.path.join(output_path, 'similarity_dict.pkl')
    cells_path = os.path.join(base, region, 'pre_ids.csv')
    shuffle_lcs_experiment.main(collapsed_path, similarity_path, cells_path,
                                output_path, 'original', num_trials=1,
                                offset=0, verbose=True)
    shuffle_lcs_experiment.main(collapsed_path, similarity_path, cells_path,
                                output_path, 'random', num_trials=30,
                                offset=1, verbose=True)
    
    # Step 4: Make histograms (for now, just the original and the random shuffles)
    figure_output_path = os.path.join(figures_base, region, experiment_name)
    os.makedirs(figure_output_path, exist_ok=True)
    make_lcs_histograms.main(output_path, figure_output_path, ['random'],
                             zscore='random', unique=False, stats=True)
    make_lcs_histograms.main(output_path, figure_output_path, ['random'],
                                zscore='random', unique=True, stats=True)
    make_klcs_histograms.main(output_path, figure_output_path, ['random'],
                                zscore='random', unique=False, stats=True)
    make_klcs_histograms.main(output_path, figure_output_path, ['random'],
                                zscore='random', unique=True, stats=True)
    
    xlim = [0.1, 200]
    make_sequencedist_distributions.main(output_path, figure_output_path, ['random'],
                                         xlim, pdf=True, cdf=True, stats=True)