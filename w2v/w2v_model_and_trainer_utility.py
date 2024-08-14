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

# utility functions for skipgram negative sampling w2v training 
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import pickle
import numpy as np
import pandas as pd

class SkipGramNS(nn.Module):
    # pytorch implementation of SGNS w2v model (Mikolov et al. 2013) with modifications
    def __init__(self, vocab_size, embedding_dim, dtype=torch.float32):
        super(SkipGramNS, self).__init__()
        self.dtype = dtype
        self.input_embedding = nn.Embedding(vocab_size, embedding_dim).to(dtype)
        self.output_embedding = nn.Embedding(vocab_size, embedding_dim).to(dtype)

        initialisation_range = 0.5 / embedding_dim
        nn.init.uniform_(self.input_embedding.weight.data, -initialisation_range, initialisation_range)
        nn.init.uniform_(self.output_embedding.weight.data, -initialisation_range, initialisation_range)

    def forward(self, input_pos, output_pos, output_neg):
        input_v = self.input_embedding(input_pos) # (batch_size * embedding_dim)
        pos_output_v = self.output_embedding(output_pos) # (batch_size * embedding_dim)
        neg_output_m = self.output_embedding(output_neg) # (batch_size * sgns_k * embedding_dim)

        pos_sim_score = torch.sum(torch.mul(pos_output_v, input_v), dim=1) # (batch_size)
        pos_sim_score = torch.clamp(pos_sim_score, max=10, min=-10)
        pos_sim_score = F.logsigmoid(pos_sim_score)

        input_v_unsqueezed = input_v.unsqueeze(2) # (batch_size * embedding_dim) -> (batch_size * embedding_dim * 1)
        neg_sim_score = torch.bmm(neg_output_m, input_v_unsqueezed).squeeze(2) # (batch_size * sgns_k * 1) -> (batch_size * sgns_k)
        neg_sim_score = torch.clamp(neg_sim_score, max=10, min=-10)
        neg_sim_score = torch.sum(F.logsigmoid(-neg_sim_score), dim=1) # (batch_size)
        
        loss = -torch.mean(pos_sim_score + neg_sim_score) # convert score to loss
        return loss

def tokenise_text(text, word_to_idx_dict) -> int:
    return [word_to_idx_dict[str(text)]][0]

def get_tokenised_text(token, idx_to_word_dict) -> str:
    return [idx_to_word_dict[str(token)]][0]

def similar_to_many_evaluation_metrics(
    embedding, word_to_idx_dict, idx_to_word_dict, inputs, top_n=10
):
    score = 0
    inputs_idx_list = [tokenise_text(input, word_to_idx_dict) for input in inputs]
    similarity_matrix = compute_similarity_matrix(embedding)
    for idx in inputs_idx_list:
        top_n_similar_words_idx_list = top_n_similar_words(
            get_tokenised_text(str(idx), idx_to_word_dict),
            word_to_idx_dict,
            idx_to_word_dict,
            similarity_matrix,
            top_n=top_n,
            to_idx_list=True,
        )
        for input_idx in inputs_idx_list:
            if input_idx in top_n_similar_words_idx_list:
                score += 1
    score = score - len(inputs)
    return score

def compute_similarity_matrix(embeddings):
    normalized_embeddings = F.normalize(
        embeddings, p=2, dim=1
    )  # v / ||v||2 across dim 1 which is word emmbedding dimension
    similarity_matrix = torch.mm(
        normalized_embeddings, normalized_embeddings.t()
    )  # V × V
    return similarity_matrix

def top_n_similar_words(
    input_word,
    word_to_idx,
    idx_to_word,
    similarity_matrix,
    top_n=10,
    to_idx_list=False,
    return_similarity=False
):
    if input_word not in word_to_idx:
        return "input word not found in vocab."

    input_idx = word_to_idx[input_word]
    input_similarities = similarity_matrix[input_idx]
    sorted_indices = torch.argsort(
        input_similarities, descending=True
    )  # torch.argsort returns indices
    top_n_indices = sorted_indices[1 : top_n + 1]

    if to_idx_list:
        top_n_words = [idx.item() for idx in top_n_indices]
    else:
        if not return_similarity:
            top_n_words = [
                idx_to_word[str(idx.item())] for idx in top_n_indices
            ]
        else:
            top_n_words = [
                (idx_to_word[str(idx.item())], input_similarities[idx].item())
                for idx in top_n_indices
            ]
    return top_n_words

def search_top_n_similar_words_to_df(
    word_list_to_search,
    top_n,
    word_to_idx_dict,
    idx_to_word_dict,
    trained_w2v_full_embedding_cosine_similarity_matrix,
) -> pd.DataFrame:
    word_list_to_search.sort()
    top_n_similar_matrix = np.zeros((top_n, len(word_list_to_search)), dtype=object)
    for i, word in enumerate(word_list_to_search):
        top_n_similar_this_word = top_n_similar_words(
            word,
            word_to_idx_dict,
            idx_to_word_dict,
            trained_w2v_full_embedding_cosine_similarity_matrix,
            top_n=top_n,
            to_idx_list=False,
        )
        top_n_similar_matrix[0:top_n, i] = top_n_similar_this_word
    top_n_similar_df = pd.DataFrame(
        top_n_similar_matrix,
        index=[f"Top {i+1}" for i in range(top_n)],
        columns=word_list_to_search,
    )
    return top_n_similar_df

def json_loader(file_path):
    return json.load(open(file_path, "r"))

def pickle_loader(file_path):
    return pickle.load(open(file_path, "rb"))

def load_resources_sgns():
    with open("../../w2v_sgns_data/w2v_data_and_vocab_config.json", "r") as f:
        data_config = json.load(f)
    with open("../../w2v_sgns_data/vocab_list.pkl", "rb") as f:
        vocab_list = pickle.load(f)
    with open("../../w2v_sgns_data/word_to_idx_dict.json", "r") as f:
        word_to_idx_dict = json.load(f)
    with open("../../w2v_sgns_data/idx_to_word_dict.json", "r") as f:
        idx_to_word_dict = json.load(f)
    with open("../../w2v_sgns_data/output_list_skipgram.pkl", "rb") as f:
        output_list = pickle.load(f)
    with open("../../w2v_sgns_data/input_list_skipgram.pkl", "rb") as f:
        input_list = pickle.load(f)
    with open("../../w2v_sgns_data/negative_samples_list_skipgram.pkl", "rb") as f:
        negative_samples = pickle.load(f)
    with open("../../w2v_sgns_data/vocab_stats_dict.pkl", "rb") as f:
        vocab_stats = pickle.load(f)
    return data_config, vocab_list, vocab_stats, word_to_idx_dict, idx_to_word_dict, input_list, output_list, negative_samples

def load_benchmark_lists():
    firm_benchm = [
        "abc_token",
        "aig_token",
        "anz_token",
        "as_token",
        "av_token",
        "axa_token",
        "az_token",
        "barc_token",
        "bbva_token",
        "bh_token",
        "blk_token",
        "bmo_token",
        "bnp_token",
        "bns_token",
        "bnym_token",
        "boa_token",
        "boc_token",
        "boca_token",
        "bocom_token",
        "boe_token",
        "boj_token",
        "bok_token",
        "bpce_token",
        "bs_token",
        "ca_token",
        "cba_token",
        "cbk_token",
        "ccb_token",
        "ceb_token",
        "cibc_token",
        "cic_token",
        "citi_token",
        "citic_token",
        "cl_token",
        "cm_token",
        "cmb_token",
        "cmbc_token",
        "cpp_token",
        "cs_token",
        "db_token",
        "dbs_token",
        "djd_token",
        "dl_token",
        "dsk_token",
        "dz_token",
        "ecb_token",
        "fed_token",
        "ff_token",
        "frs_token",
        "g_token",
        "gpif_token",
        "gs_token",
        "hdfc_token",
        "hfg_token",
        "hsbc_token",
        "iag_token",
        "ibc_token",
        "icbc_token",
        "icici_token",
        "ing_token",
        "isp_token",
        "ivz_token",
        "jpb_token",
        "jpm_token",
        "kbfg_token",
        "kkr_token",
        "lgen_token",
        "lloy_token",
        "met_token",
        "mfc_token",
        "mfg_token",
        "mqg_token",
        "mr_token",
        "ms_token",
        "mufg_token",
        "nab_token",
        "nbc_token",
        "nbim_token",
        "nbs_token",
        "nda_token",
        "nl_token",
        "nmr_token",
        "nps_token",
        "nrck_token",
        "nwrbs_token",
        "nyslrs_token",
        "ocbc_token",
        "pa_token",
        "pnc_token",
        "pru_token",
        "qbe_token",
        "rba_token",
        "rbb_token",
        "rbc_token",
        "rbi_token",
        "san_token",
        "sbi_token",
        "sc_token",
        "sfg_token",
        "sg_token",
        "smfg_token",
        "smth_token",
        "spd_token",
        "ss_token",
        "sun_token",
        "td_token",
        "tmk_token",
        "ubs_token",
        "ucg_token",
        "uhg_token",
        "uob_token",
        "usb_token",
        "uss_token",
        "wbc_token",
        "wfc_token",
        "wfg_token",
        "zig_token",
    ]
    currencies_benchm = [
        "chf",
        "gbp",
        "usd",
        "jpy",
        "cad",
        "rmb",
        "eur",
        "aud",
        "euro",
        "dollar",
        "yen",
        "libor",
        "fx",
    ]
    countries_benchm = [
        "america",
        "kingdom",
        "germany",
        "france",
        "italy",
        "china",
        "japan",
        "korea",
        "switzerland",
        "australia",
        "canada",
        "india",
    ]
    return firm_benchm, currencies_benchm, countries_benchm

def clean_up(*args):
    """Clean up memory by deleting unused data."""
    for arg in args:
        del arg 