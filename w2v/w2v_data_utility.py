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

# utility functions for generating purpose-specific skipgram negative sampling dataset 
from pathlib import Path
from collections import Counter
import json
import re
from numba import njit, prange
import numpy as np
from tqdm import tqdm

def load_abbrevs_to_names_dict(add_firm_token_suffix = True) -> dict:
    """
    load the abbrevs_to_names_dict.json file and add firm token suffix to the keys if add_firm_token_suffix is True
    """
    with open(Path("abbrevs_to_names_dict.json"), 'r') as f:
        abbrevs_to_names_dict = json.load(f)

    if add_firm_token_suffix:
        abbrevs_to_names_dict_with_token_suffix = {}
        for firm in abbrevs_to_names_dict.keys():
            abbrevs_to_names_dict_with_token_suffix[f"{firm}_token"] = abbrevs_to_names_dict[firm]
        with open(Path("abbrevs_to_names_dict_with_token_suffix.json"), 'w') as f:
            json.dump(abbrevs_to_names_dict_with_token_suffix, f, indent=4)
        return abbrevs_to_names_dict_with_token_suffix
    else:
        return abbrevs_to_names_dict
    
def compile_abbrevs_replacements(abbrevs_to_names_dict) -> dict:
    """
    compile regex patterns of firm related names for every firm for replacement
    """
    compiled_abbrevs_replacements = {}
    for abbrevs_token, firm_names in abbrevs_to_names_dict.items():
        pattern = re.compile(r'\b(' + '|'.join(firm_names) + r')\b')
        compiled_abbrevs_replacements[abbrevs_token] = pattern
    return compiled_abbrevs_replacements

def replace_firm_names_from_sentence(firm, sent, compiled_abbrevs_replacements) -> str:
    """
    replace related firm names with firm tokens in a sentence
    """
    firm_token = f"{firm}_token"
    return compiled_abbrevs_replacements[firm_token].sub(firm_token, sent)

def collect_usable_sentences(
    master_data_dict,
    time_range,
    firm_list,
    normalisation_level,
    min_sentence_len=3,
    firm_name_replaced_with_abbrevs = True,
    firm_name_replaced_with_abbrevs_global = False,
    verbose = 1) -> list:
    """
    Collect usable sentences from the master data dictionary.
    Return a single list containing all sentence
    """
    usable_sentences = []
    except_counter = 0
    if firm_name_replaced_with_abbrevs:
        abbrevs_to_names_dict = load_abbrevs_to_names_dict(add_firm_token_suffix = True)
        compiled_abbrevs_replacements = compile_abbrevs_replacements(abbrevs_to_names_dict)
        for firm in tqdm(firm_list, desc="Processing Firms"):
            for year in time_range:
                try:
                    for page in master_data_dict[firm][str(year)][normalisation_level]:
                        for sent in page:
                            if len(sent.split(" ")) > min_sentence_len:
                                sent = replace_firm_names_from_sentence(firm, sent, compiled_abbrevs_replacements)
                                usable_sentences.append(sent)
                except (KeyError, TypeError):
                    except_counter += 1
                    pass
        if firm_name_replaced_with_abbrevs_global:
            usable_sentences_global_name_replaced = []
            for sent in tqdm(usable_sentences, desc="Processing Sentences"):
                for firm in firm_list:
                    sent = replace_firm_names_from_sentence(firm, sent, compiled_abbrevs_replacements)
                usable_sentences_global_name_replaced.append(sent)
            if verbose != 0:
                print(f"Number of missing years: {except_counter}")
                print(f"Number of usable sentences: {len(usable_sentences_global_name_replaced)}")
            return usable_sentences_global_name_replaced
        else:
            if verbose != 0:
                print(f"Number of missing years: {except_counter}")
                print(f"Number of usable sentences: {len(usable_sentences)}")
            return usable_sentences
    else:
        for firm in tqdm(firm_list, desc="Processing Firms"):
            for year in time_range:
                try:
                    for page in master_data_dict[firm][str(year)][normalisation_level]:
                        for sent in page:
                            if len(sent.split(" ")) > min_sentence_len:
                                usable_sentences.append(sent)
                except (KeyError, TypeError):
                    except_counter += 1
                    pass
        if verbose != 0:
            print(f"Number of missing years: {except_counter}")
            print(f"Number of usable sentences: {len(usable_sentences)}")
        return usable_sentences

def generate_ngram(text_list, n_range=(1, 1)) -> list:
    """
    Generate ngram from a list of text.
    """
    ngrams = []
    low_n_bound, up_n_bound = n_range

    if up_n_bound == low_n_bound:
        for i in range(len(text_list) - up_n_bound + 1):
            ngrams.append(" ".join(text_list[i : i + up_n_bound]))

    if up_n_bound > low_n_bound:
        for n in range(1, up_n_bound + 1):
            ngram = []
            for i in range(len(text_list) - n + 1):
                ngram.append(" ".join(text_list[i : i + n]))
            ngrams = ngrams + ngram

    return ngrams

def generate_ngram_to_position(text_list, n_range=(1, 1)) -> list:
    """
    Generate ngram from a list of text. The generated ngram will be placed after the word used to generat that ngram.
    """
    ngrams = []
    low_n_bound, up_n_bound = n_range
    if type(text_list) == str:
        text_list = text_list.split(" ")
    else:
        text_list = text_list

    if up_n_bound == low_n_bound:
        for i in range(len(text_list) - up_n_bound + 1):
            ngrams.append(" ".join(text_list[i : i + up_n_bound]))

    if up_n_bound > low_n_bound:
        for i in range(len(text_list) - up_n_bound + 1):
            for n in range(1, up_n_bound + 1):
                ngrams.append(" ".join(text_list[i : i + n]))
    return ngrams

def generate_ngram_dataset(sentence_list, to_position = True, n_range=(1,1)) -> list:
    """
    Generate ngram dataset
    """
    ngram_dataset = []
    if to_position:
        for sent in sentence_list:
            ngram_dataset.append(generate_ngram_to_position(sent, n_range=n_range))
    else:
        for sent in sentence_list:
            ngram_dataset.append(generate_ngram(sent, n_range=n_range))
    return ngram_dataset


def calculate_frequency_and_proportion(text_list: list, verbose=0) -> dict:
    """
    Calculate pre_vocab_frequency and pre_vocab_proportion of each text(token) in the corpus.
    """
    all_text_list = [word for sent in text_list for word in sent]
    total_text_count = len(all_text_list)
    unique_text_list = sorted(list(set(all_text_list)))

    if verbose !=0:
        print("Total number of tokens: ",total_text_count)
        print("Total number of unique tokens: ",len(unique_text_list))

    unique_text_list_with_freq_prop_dict = {}
    counter = Counter(all_text_list)
    for text in unique_text_list:
        current_text_count = counter[text]
        unique_text_list_with_freq_prop_dict[text] = {}
        unique_text_list_with_freq_prop_dict[text]["pre_vocab_frequency"] = current_text_count
        unique_text_list_with_freq_prop_dict[text]["pre_vocab_proportion"] = round(current_text_count / total_text_count, 8)
    
    return unique_text_list_with_freq_prop_dict

def calculate_rank(unique_text_list_with_freq_prop_dict: dict) -> dict:
    """
    Calculate the rank of each word in the corpus.
    """
    sorted_text_list = sorted(unique_text_list_with_freq_prop_dict.items(), key=lambda item: item[1]['pre_vocab_frequency'], reverse=True)
    
    current_rank = 0
    last_freq = 0
    for i, (text, item) in enumerate(sorted_text_list):
        if last_freq == 0 or last_freq > item["pre_vocab_frequency"]:
            current_rank = i + 1
        last_freq = item["pre_vocab_frequency"]
        unique_text_list_with_freq_prop_dict[text]["rank"] = current_rank
        
    return unique_text_list_with_freq_prop_dict

def create_vocab(unique_text_list_with_freq_prop_dict, min_freq=0, max_freq=None, max_n_top_features=None, create_oov_token=False, verbose=0):
    """
    Create the vocabulary
    """
    unique_text_list_with_freq_prop_dict = calculate_rank(unique_text_list_with_freq_prop_dict)
    vocab_list = []
    vocab_stats_dict = {}
    for text, item in unique_text_list_with_freq_prop_dict.items():
        if max_freq is not None and item["pre_vocab_frequency"] > min_freq:
            if max_freq > item["pre_vocab_frequency"]:
                vocab_list.append(text)
                vocab_stats_dict[text] = item
        else:
            if item["pre_vocab_frequency"] > min_freq:
                vocab_list.append(text)
                vocab_stats_dict[text] = item

    if max_n_top_features is not None:
        # if many text have the same rank, and cutoff is inbetween, cutoff will be performed at ramdom within that rank
        sorted_filtered_text_list = sorted(vocab_stats_dict.items(), key=lambda item: item[1]["rank"])
        vocab_list = [sorted_filtered_text_list[i][0] for i in range(0,len(sorted_filtered_text_list))][:max_n_top_features]
        vocab_stats_dict = {}
        for text in vocab_list:
            vocab_stats_dict[text] = unique_text_list_with_freq_prop_dict[text]

    if create_oov_token:
        vocab_list.append("<OOV>")
        vocab_stats_dict["<OOV>"] = {}
        vocab_stats_dict["<OOV>"]["pre_vocab_frequency"] = -1
        vocab_stats_dict["<OOV>"]["pre_vocab_proportion"] = -0.0
        vocab_stats_dict["<OOV>"]["rank"] = -1

    word_to_idx_dict = {word:idx for idx, word in enumerate(vocab_list)}
    idx_to_word_dict = {idx:word for idx, word in enumerate(vocab_list)}
    # can aldo return: unique_text_list_with_freq_prop_dict (this is dict of all text without any filter)
    if verbose != 0:
        print("Vocabulary size: ", len(vocab_list))

    return word_to_idx_dict, idx_to_word_dict, vocab_list, vocab_stats_dict

def tokenise_text(text, word_to_idx_dict) -> int:
    return [word_to_idx_dict[text]][0]

def get_tokenised_text(token, idx_to_word_dict) -> str:
    return [idx_to_word_dict[token]][0]

def remove_oov_from_sentences(sentence_list, word_to_idx_dict) -> list:
    """
    remove oov token from the sentences 
    """
    sentence_list_with_no_oov = []
    for sent in sentence_list:
        sentence_with_no_oov = []
        for words in sent:
            if words in word_to_idx_dict:
                sentence_with_no_oov.append(words)
        sentence_list_with_no_oov.append(sentence_with_no_oov)
    return sentence_list_with_no_oov

def create_skipgram_dataset(sentence_list, word_to_idx_dict, half_context_size=3,return_quali_sent_used= True, verbose=0):
    """
    Create the Skipgram dataset with windows size = half_context_size* 2 + 1.
    Sentences with length smaller than windows size will be removed.
    Input is generated_ngram_dataset.
    """
    input_list = []
    output_list = []
    sentence_used_to_create_dataset = []
    for sentence in sentence_list:
        if word_to_idx_dict is not None:
            if len(sentence) < half_context_size * 2 + 1:
                pass
            else:
                if return_quali_sent_used:
                    sentence_used_to_create_dataset.append(sentence)
                for i in range(half_context_size, len(sentence) - half_context_size):
                    for j in range(1, half_context_size + 1):
                        input_list.append(tokenise_text(sentence[i], word_to_idx_dict))
                        output_list.append(tokenise_text(sentence[i - j], word_to_idx_dict))
                        input_list.append(tokenise_text(sentence[i], word_to_idx_dict))
                        output_list.append(tokenise_text(sentence[i + j], word_to_idx_dict))
        else:
            if len(sentence) < half_context_size * 2 + 1:
                pass
            else:
                if return_quali_sent_used:
                    sentence_used_to_create_dataset.append(sentence)
                for i in range(half_context_size, len(sentence) - half_context_size):
                    for j in range(1, half_context_size + 1):
                        input_list.append(sentence[i])
                        output_list.append(sentence[i - j])
                        input_list.append(sentence[i])
                        output_list.append(sentence[i + j])
    if verbose != 0:
        print("Window size: ", half_context_size * 2 + 1)
        print("Number of context sentence for training: ", len(input_list))
    return input_list, output_list, sentence_used_to_create_dataset

def calculate_post_vocab_proportion(vocab_stats_dict: dict, sentence_used_to_create_dataset: list) -> dict:
    all_text_list = [word for sent in sentence_used_to_create_dataset for word in sent]
    total_text_count = len(all_text_list)
    unique_text_list = sorted(list(set(all_text_list)))

    counter = Counter(all_text_list)
    for text in unique_text_list:
        current_text_count = counter[text]
        vocab_stats_dict[text]["post_vocab_frequency"] = current_text_count
        vocab_stats_dict[text]["post_vocab_proportion"] = round(current_text_count / total_text_count, 8)
    return vocab_stats_dict

def distribution_scaling(vocab_stats_dict, power=0.75):
    
    for item in vocab_stats_dict.keys():
        vocab_stats_dict[item]["post_vocab_frequency_scaled"] = vocab_stats_dict[item]["post_vocab_frequency"] ** power
    
    total_scaled_post_vocab_frequency = sum([vocab_stats_dict[item]["post_vocab_frequency_scaled"] for item in vocab_stats_dict.keys()])
    
    for item in vocab_stats_dict.keys():
        vocab_stats_dict[item]["post_vocab_proportion_scaled"] = vocab_stats_dict[item]["post_vocab_frequency_scaled"] / total_scaled_post_vocab_frequency
    
    return vocab_stats_dict

def find_all_positive_negative_output_idx(output_list_skipgram, input_list_skipgram, vocab_stats_dict, idx_to_word_dict, word_to_idx_dict, with_self = False):
    """
    Find all pos and negative output of every token based on the skipgram dataset.
    Two keys will be added to the vocab_stats_dict: "pos_output_set" and "neg_output_set", 
    storing unique positive and negative output idx for that token.
    """
    all_vocab_list = vocab_stats_dict.keys()
    
    for item in all_vocab_list:
        vocab_stats_dict[item]["pos_output_set"] = set()

    for output_idx, input_idx in zip(output_list_skipgram, input_list_skipgram):
        vocab_stats_dict[idx_to_word_dict[output_idx]]["pos_output_set"].add(input_idx)
        vocab_stats_dict[idx_to_word_dict[input_idx]]["pos_output_set"].add(output_idx)
    
    all_vocab_idx = idx_to_word_dict.keys()
    for i_idx in all_vocab_idx:
        vocab_stats_dict[idx_to_word_dict[i_idx]]["neg_output_set"] = set()
        
        if with_self:
            # If with_self is True, then the negative output will include the word itself.
            vocab_stats_dict[idx_to_word_dict[i_idx]]["neg_output_set"].add(i_idx)

        for j in all_vocab_list:
            if i_idx not in vocab_stats_dict[j]["pos_output_set"]:
                # add words that not appeared togather with positive output as negative output
                vocab_stats_dict[idx_to_word_dict[i_idx]]["neg_output_set"].add(word_to_idx_dict[j])
    return vocab_stats_dict

def negative_sampling_using_numba_data_prep(vocab_stats_dict, idx_to_word_dict, input_list_skipgram):
    """
    prepare data for negative sampling using numba multiprocessing
    """
    neg_output_arrays = {}
    for idx, word in idx_to_word_dict.items():
        neg_output_arrays[idx] = np.array(list(vocab_stats_dict[word]["neg_output_set"]))
    neg_output_indices = list(neg_output_arrays.keys())
    neg_output_data = list(neg_output_arrays.values())
    input_list_skipgram_array = np.array(input_list_skipgram)

    return neg_output_indices, neg_output_data, input_list_skipgram_array

@njit(nogil=True, parallel=True)
def create_skipgram_negative_samples_from_uniform_dist_numba(neg_output_indices, neg_output_data, input_list_skipgram_array, k=2):
    """
    Negative output samples are sampled based on the uniform distribution of input word's negative words, i.e., words that have not 
    appeared around of the input word before.
    """
    # Pre-allocate array for results
    num_samples = len(input_list_skipgram_array)
    negative_samples_list = np.empty((num_samples, k), dtype=np.int32)
    
    # Use prange for parallel loop
    for i in prange(num_samples):
        input_idx = input_list_skipgram_array[i]
        output_data_idx = neg_output_indices[input_idx]
        output_data = neg_output_data[output_data_idx]
        
        # Perform the sampling
        choices = np.random.choice(output_data, k, replace=False)
        negative_samples_list[i] = choices

    return negative_samples_list