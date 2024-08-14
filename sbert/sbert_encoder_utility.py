# utility functions for sbert encoder
import json
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def collect_sentences_to_data_dict(firm_list, time_range, normalisation_level) -> dict:
    """
    Collect all sentences from the parsed PDF files to a dictionary
    """
    data_dict = {}
    for firm in tqdm(firm_list):
        data_dict[firm] = {}
        firm_level_retrieval_path = (
            Path("../../preprocessed_and_parsed_report_data") / f"{firm}_data.json"
        )
        with open(firm_level_retrieval_path, "r") as file:
            firm_dict = json.load(file)

        for year in time_range:
            data_dict[firm][str(year)] = {}
            try:
                data_dict[firm][str(year)][normalisation_level] = firm_dict[firm][
                    str(year)
                ][normalisation_level]
            except TypeError:
                pass
    return data_dict

def filter_usable_sentences_to_data_dict(
    data_dict, time_range, firm_list, normalisation_level, min_sentence_len=3, verbose=1
) -> dict:
    """
    Create a new key in the data_dict for each firm each year
    """
    usable_sentences_count = 0
    except_counter = 0
    for firm in tqdm(firm_list, desc="Processing Firms"):
        for year in time_range:
            usable_sentences = []
            try:
                for page in data_dict[firm][str(year)][normalisation_level]:
                    for sent in page:
                        if len(sent.split(" ")) > min_sentence_len:
                            usable_sentences.append(sent)
                data_dict[firm][str(year)][
                    f"{normalisation_level}_usable_sentences"
                ] = usable_sentences
            except (KeyError, TypeError):
                except_counter += 1
                pass
            usable_sentences_count += len(usable_sentences)
    if verbose != 0:
        print(f"Number of missing years: {except_counter}")
        print(f"Number of usable sentences: {usable_sentences_count}")
    return data_dict

def embed_sentence_to_data_dict(
    data_dict,
    pretrained_sbert_models="all-MiniLM-L6-v2",
    normalisation_level="docs_level_sentence_list_text_normalised",
    verbose=1,
) -> dict:
    '''
    Encode sentences to SBERT embeddings
    '''
    model = SentenceTransformer(pretrained_sbert_models)
    except_counter = 0
    firm_with_missing_data = []
    for firm in tqdm(data_dict.keys(), desc="Processing Firms"):
        try:
            for year in data_dict[firm].keys():
                sentences = data_dict[firm][year][
                    f"{normalisation_level}_usable_sentences"
                ]
                s_embeddings = model.encode(sentences)
                data_dict[firm][year][
                    f"{normalisation_level}_usable_sentences_sbert_embeddings"
                ] = s_embeddings
        except KeyError:
            except_counter += 1
            firm_with_missing_data.append(firm)
    if verbose != 0:
        print(f"Number of missing years: {except_counter}")
        print(f"Firms with missing data:", firm_with_missing_data)
    return data_dict