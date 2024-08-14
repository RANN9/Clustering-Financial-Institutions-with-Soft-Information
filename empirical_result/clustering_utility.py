# Copyright 2024 RAN
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# utility functions for statistical and network clustering
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
from scipy.stats import pearsonr, spearmanr, kendalltau

def extract_yearly_gsib_data_from_excel(file_path, start_year, end_year) -> dict:
    """
    extract G-SIB data from excel
    """
    indicator_score_dict = {}
    for year in range(start_year, end_year + 1):
        sheet_name = f"{year}_score"
        df = pd.read_excel(file_path, sheet_name=sheet_name, index_col='firm')
        df.drop(columns='no', inplace=True)
        df.rename(columns={col: col + f'_{year}' for col in df.columns}, inplace=True)
        indicator_score_dict[year] = df
    return indicator_score_dict

def merge_yearly_gsib_data_to_df(yearly_dfs)->pd.DataFrame:
    """
    merge yearly G-SIB data to a single dataframe
    """
    merged_df = pd.concat(yearly_dfs.values(), axis=1)
    return merged_df

def compute_cosine_similarity_matrix(embeddings) -> np.ndarray:
    """
    Compute cosine similarity matrix from embedding.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    similarity_matrix = np.clip(similarity_matrix, -1, 1)
    np.fill_diagonal(similarity_matrix, "NaN")
    return similarity_matrix

def create_weighted_graph_from_df(sim_or_corr_matrix_df)-> nx.Graph:
    """
    Create a weighted graph from a correlation matrix.
    """
    G_weighted = nx.Graph()
    firm_list = list(sim_or_corr_matrix_df.columns)
    G_weighted.add_nodes_from(firm_list)
    for i in firm_list:
        for j in firm_list:
            if i != j:
                G_weighted.add_edge(i, j, weight=sim_or_corr_matrix_df.loc[i, j])
    return G_weighted

def exp_scale_to_df(matrix, firm_list) -> pd.DataFrame:
    """
    Exponential scaling of a matrix.
    """
    scaled_matrix = np.exp(matrix)
    scaled_matrix_df = pd.DataFrame(scaled_matrix, columns=firm_list, index=firm_list)
    np.fill_diagonal(scaled_matrix, "NaN")
    return scaled_matrix_df

def calculate_mean_sbert_each_year_each_firm(data_dict, firm_list, time_range, normalisation_level="docs_level_sentence_list_text_normalised") -> dict:
    """
    calculate the mean of SBERT embeddings for each year of each firm
    """
    for firm in tqdm(firm_list):
        for year in time_range:
            data_dict[firm][str(year)]['yearly_average_sbert_embeddings'] = np.mean(data_dict[firm][str(year)][f"{normalisation_level}_usable_sentences_sbert_embeddings"], axis=0)
    return data_dict

def calculate_mean_firm_level_embedding(time_range, firm_list, data_dict, num_dimension) -> dict:
    """ 
    Calculate the mean of embedding across time for each firm and store in a dictionary
    """
    firm_level_mean_embedding_dict = {}
    for firm in firm_list:
        all_time_sbert = np.zeros((len(time_range), num_dimension))
        for i, years in enumerate(time_range):
            all_time_sbert[i,:] = data_dict[firm][str(years)]['yearly_average_sbert_embeddings']
        firm_level_mean_embedding_dict[firm] = np.mean(all_time_sbert, axis=0)
    return firm_level_mean_embedding_dict

def collect_mean_firm_level_embedding_to_matrix(firm_list, firm_level_mean_embedding_dict, num_dimension) -> np.ndarray:
    """ 
    collect each firm's mean embedding into a matrix of shape (num_firm, num_dimension)
    """
    firm_level_mean_embedding_dict_matrix = np.zeros((len(firm_list), num_dimension))
    for i, firm in enumerate(firm_list):
        firm_level_mean_embedding_dict_matrix[i,:] = firm_level_mean_embedding_dict[firm]
    return firm_level_mean_embedding_dict_matrix

# ----- for permutation test ----- 
# utility functions for testing the significance of correlation between two matrices

def convert_to_distance_matrix(matrix):
    """
    simple distance conversion for matrix with negative values
    """
    dist_matrix = 1 - matrix
    np.fill_diagonal(dist_matrix, 0)
    return dist_matrix

def flatten_upper_triangle(matrix):
    """
    obtain the upper triangle of a matrix and convert it to a 1D vector
    """
    triu_indices_mask = np.triu_indices(matrix.shape[0], k=1)
    return matrix[triu_indices_mask]

def significance_stars(p_value):
    """
    determine the significance level of p-value
    """
    if p_value <= 0.001:
        return '***'
    elif p_value <= 0.01:
        return '**'
    elif p_value <= 0.05:
        return '*'
    else:
        return ''

def permutation_test(matrix_x, matrix_y, convert_to_dist = True, num_permutations = 9999, correlation_type = 'pearson', round_decimal = 8, seed = 9):
    """
    Perform permutation test (two sided) to assess the significance of the correlation between two matrices.
    Simple distance conversion will be applied if convert_to_dist is True. (usually dose not affect the result)

    Returns:
    - r_obs: float
        Observed correlation coefficient between the two distance matrices
    - p_value: float
        p-value indicating the significance of the observed correlation
    - significance: str
        significance level of the p-value
    - r_perm_list: list
        list of permuted correlation coefficients
        
    Reference: 
    - Mantel, N. (1967). The detection of disease clustering and a generalized regression approach
    - https://github.com/vegandevs/vegan/blob/master/R/mantel.R
    """
    np.random.seed(seed)

    if convert_to_dist:
        matrix_x = convert_to_distance_matrix(matrix_x)
        matrix_y = convert_to_distance_matrix(matrix_y)
    else:
        np.fill_diagonal(matrix_x, 1)
        np.fill_diagonal(matrix_y, 1)

    # Choose correlation function
    if correlation_type == 'pearson':
        corr_func = pearsonr
    if correlation_type == 'spearman':
        corr_func = spearmanr
    if correlation_type == 'kendalltau':
        corr_func = kendalltau
    
    # Compute observed correlation
    r_obs, _ = corr_func(flatten_upper_triangle(matrix_x), flatten_upper_triangle(matrix_y))
    
    # Perform permutations
    N = matrix_x.shape[0]
    matrix_idx = np.arange(N)
    r_perm_list = []
    for _ in range(num_permutations):
        permuted_idx = np.random.permutation(matrix_idx)
        permuted_matrix_x = matrix_x[permuted_idx][:, permuted_idx]
        r_perm, _ = corr_func(flatten_upper_triangle(permuted_matrix_x), flatten_upper_triangle(matrix_y))
        r_perm_list.append(r_perm)
    
    count = np.sum(np.abs(r_perm_list) >= np.abs(r_obs))
    p_value = (count + 1) / (num_permutations + 1)

    return round(r_obs, round_decimal), round(p_value, round_decimal), significance_stars(p_value), r_perm_list