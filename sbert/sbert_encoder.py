from sbert_encoder_utility import *
import os
import pickle
import json
from pathlib import Path

time_range = list(range(2013,2023))
min_sentence_len_to_collect = 5
firm_list = os.listdir(Path("../../report_data"))
try:
    firm_list.remove(".DS_Store") # ensure code robustness for MacOS
except:
    pass
firm_list.sort()

## --------choose different sbert encoder models--------
# SBERT_model_metadata_all_MiniLM_L6_v2 = {
#     "model": "all-MiniLM-L6-v2",
#     "max_seq_length": 256,
#     "num_dimension": 384}
# with open(Path(f"../../embedding_data/sbert/SBERT_model_metadata_all_MiniLM_L6_v2.json"), "w") as file:
#     json.dump(SBERT_model_metadata_all_MiniLM_L6_v2, file)
SBERT_model_metadata_all_mpnet_base_v2 = {
    "model": "all-mpnet-base-v2",
    "max_seq_length": 384,
    "num_dimension": 768}
with open(Path(f"../../embedding_data/sbert/SBERT_model_metadata_all_mpnet_base_v2.json"), "w") as file:
    json.dump(SBERT_model_metadata_all_mpnet_base_v2, file)

## --------encode text normalised data--------
text_normalised_data_dict = collect_sentences_to_data_dict(
    firm_list, 
    time_range, 
    "docs_level_sentence_list_text_normalised")
text_normalised_data_dict = filter_usable_sentences_to_data_dict(
    text_normalised_data_dict,
    time_range,
    firm_list,
    "docs_level_sentence_list_text_normalised",
    min_sentence_len=min_sentence_len_to_collect,
    verbose = 1)
sbert_encoded_text_normalised_data_all_mpnet_base_v2 = embed_sentence_to_data_dict(
    text_normalised_data_dict,
    pretrained_sbert_models=SBERT_model_metadata_all_mpnet_base_v2["model"],
    normalisation_level="docs_level_sentence_list_text_normalised")
with open(Path(f"../../embedding_data/sbert/sbert_encoded_text_normalised_data_all_mpnet_base_v2.pkl"), "wb") as file:
    pickle.dump(sbert_encoded_text_normalised_data_all_mpnet_base_v2, file)
del text_normalised_data_dict, sbert_encoded_text_normalised_data_all_mpnet_base_v2

## --------encode text and entity normalised data--------
# text_and_ent_normalised_data_dict = collect_sentences_to_data_dict(
#     firm_list, 
#     time_range, 
#     "docs_level_sentence_list_text_and_ent_normalised")
# text_and_ent_normalised_data_dict = filter_usable_sentences_to_data_dict(
#     text_and_ent_normalised_data_dict,
#     time_range,
#     firm_list,
#     "docs_level_sentence_list_text_and_ent_normalised",
#     min_sentence_len=min_sentence_len_to_collect,
#     verbose = 1)
# sbert_encoded_text_and_ent_normalised_data_all_mpnet_base_v2 = embed_sentence_to_data_dict(
#     text_and_ent_normalised_data_dict,
#     pretrained_sbert_models=SBERT_model_metadata_all_mpnet_base_v2["model"],
#     normalisation_level="docs_level_sentence_list_text_and_ent_normalised")
# with open(Path(f"../../embedding_data/sbert/sbert_encoded_text_and_ent_normalised_data_all_mpnet_base_v2.pkl"), "wb") as file:
#     pickle.dump(sbert_encoded_text_and_ent_normalised_data_all_mpnet_base_v2, file)
# del text_and_ent_normalised_data_dict, sbert_encoded_text_and_ent_normalised_data_all_mpnet_base_v2
