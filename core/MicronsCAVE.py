from caveclient import CAVEclient
import numpy as np
import pandas as pd
from time import sleep
from requests.exceptions import HTTPError
from tqdm import tqdm

class CAVE:

    @staticmethod
    def get_cell_type(pt_root_id, cell_df):
        try:
            return cell_df[cell_df['pt_root_id'] == pt_root_id]['cell_type'].to_list()[0]
        except KeyError:
            return "Unknown"
        except IndexError:
            return "Unknown"
        
    @staticmethod
    def rescale_position(position):
        scale_vector = np.array([4/1000, 4/1000, 40/1000])
        return position * scale_vector

    def __init__(self, version='v343'):
        self.version = version
        self.client = CAVEclient(f'minnie65_public_{version}')

    def download_cells(self, filter_dict):
        cell_df = self.client.materialize.query_table('aibs_soma_nuc_metamodel_preds_v117',
                                                      filter_in_dict = filter_dict,
                                                      select_columns=['pt_root_id', 'cell_type', 'pt_position'])
        
        cell_df['pt_position'] = cell_df['pt_position'].apply(CAVE.rescale_position)
        position_df = cell_df['pt_position'].apply(pd.Series)
        position_df.columns = ['pt_x', 'pt_y', 'pt_z']

        cell_df = cell_df.drop('pt_position', axis=1)
        cell_df = pd.concat([cell_df, position_df], axis=1)

        # remove rows with identical pt_root_id
        cell_df = cell_df.drop_duplicates(subset=['pt_root_id'])
        return cell_df

    def download_excitatory_cells(self):
        filter_dict = {'cell_type': ['23P', '4P', '5P-IT', '5P-ET', '5P-NP', '6P-IT', '6P-CT']}
        return self.download_cells(filter_dict)

    def download_inhibitory_cells(self):
        filter_dict = {'cell_type': ['BC', 'MC', 'BPC', 'NGC']}
        return self.download_cells(filter_dict)

    def download_synapses(self, filter_dict, cell_df=None):

        syn_df = self.client.materialize.query_table('synapses_pni_2',
                                                    filter_in_dict = filter_dict,
                                                    select_columns=['id', 'pre_pt_root_id', 'post_pt_root_id', 'ctr_pt_position', 'size'])
        
        if cell_df is not None:
            syn_df['cell_type_pre'] = syn_df['pre_pt_root_id'].apply(lambda x: CAVE.get_cell_type(x, cell_df))
            syn_df['cell_type_post'] = syn_df['post_pt_root_id'].apply(lambda x: CAVE.get_cell_type(x, cell_df))
        else:
            syn_df['cell_type_pre'] = 'Unknown'
            syn_df['cell_type_post'] = 'Unknown'
        
        syn_df['ctr_pt_position'] = syn_df['ctr_pt_position'].apply(CAVE.rescale_position)
        position_df = syn_df['ctr_pt_position'].apply(pd.Series)
        position_df.columns = ['ctr_pt_x', 'ctr_pt_y', 'ctr_pt_z']

        # syn_df = syn_df.drop('ctr_pt_position', axis=1)
        syn_df = pd.concat([syn_df, position_df], axis=1)

        return syn_df
    
    def download_input_synapses(self, post_pt_root_ids, cell_df=None):
        if type(post_pt_root_ids) == int:
            post_pt_root_ids = [post_pt_root_ids]

        filter_dict = {'post_pt_root_id': post_pt_root_ids}
        syn_df = self.download_synapses(filter_dict, cell_df)

        if len(syn_df) >= 500000:
            chunk_1 = post_pt_root_ids[:len(post_pt_root_ids)//2]
            filter_dict_1 = {'post_pt_root_id': chunk_1}
            chunk_2 = post_pt_root_ids[len(post_pt_root_ids)//2:]
            filter_dict_2 = {'post_pt_root_id': chunk_2}
            syn_df_1 = self.download_synapses(filter_dict_1, cell_df)
            syn_df_2 = self.download_synapses(filter_dict_2, cell_df)
            syn_df = pd.concat([syn_df_1, syn_df_2], axis=0)

        return syn_df

    def download_output_synapses(self, pre_pt_root_ids, cell_df=None):
        if type(pre_pt_root_ids) == int:
            pre_pt_root_ids = [pre_pt_root_ids]

        filter_dict = {'pre_pt_root_id': pre_pt_root_ids}
        syn_df = self.download_synapses(filter_dict, cell_df)

        if len(syn_df) >= 500000:
            chunk_1 = pre_pt_root_ids[:len(pre_pt_root_ids)//2]
            filter_dict_1 = {'pre_pt_root_id': chunk_1}
            chunk_2 = pre_pt_root_ids[len(pre_pt_root_ids)//2:]
            filter_dict_2 = {'pre_pt_root_id': chunk_2}
            syn_df_1 = self.download_synapses(filter_dict_1, cell_df)
            syn_df_2 = self.download_synapses(filter_dict_2, cell_df)
            syn_df = pd.concat([syn_df_1, syn_df_2], axis=0)
        
        return syn_df

    def download_input_synapses_list(self, post_pt_root_ids, cell_df=None, timeout=600, chunk_size=150):
        num_chunks = int(np.ceil((len(post_pt_root_ids))/chunk_size))
        for chunk in tqdm(range(num_chunks)):
            chunk_ids = post_pt_root_ids[chunk*chunk_size:(chunk+1)*chunk_size]
            try:
                syn_df = self.download_input_synapses(chunk_ids, cell_df)
            except HTTPError:
                print(f"Chunk {chunk} failed, retrying")
                sleep(timeout)
                syn_df = self.download_input_synapses(chunk_ids, cell_df)
            synapses_grouped = syn_df.groupby('post_pt_root_id')
            yield synapses_grouped

    def download_output_synapses_list(self, pre_pt_root_ids, cell_df=None, timeout=600, chunk_size=750):
        num_chunks = int(np.ceil((len(pre_pt_root_ids))/chunk_size))
        for chunk in tqdm(range(num_chunks)):
            chunk_ids = pre_pt_root_ids[chunk*chunk_size:(chunk+1)*chunk_size]
            try:
                syn_df = self.download_output_synapses(chunk_ids, cell_df)
            except HTTPError:
                print(f"Chunk {chunk} failed, retrying")
                sleep(timeout)
                syn_df = self.download_output_synapses(chunk_ids, cell_df)
            synapses_grouped = syn_df.groupby('pre_pt_root_id')
            yield synapses_grouped