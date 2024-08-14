# preprocess and parse pdf files for various NLP tasks
import os
import json
from tqdm import tqdm
import spacy
from pathlib import Path
from preprocess_and_parse_utility import *

# initialise spacy model
spacy.require_gpu()
nlp = spacy.load('en_core_web_trf')
nlp.max_length = 1000000000

# initialise entity labels to remove and not to concate
ent_labels_to_remove  = ["CARDINAL", "DATE", "EVENT", "FAC", "GPE", "LANGUAGE", "LAW", 
                                    "LOC", "MONEY", "NORP", "ORDINAL", "ORG", "PERCENT", "PERSON", 
                                    "PRODUCT", "QUANTITY", "TIME", "WORK_OF_ART"]
ent_labels_not_to_concate = ["DATE", "CARDINAL", "PERCENT", "TIME"]

# load british to american english dictionary 
with open("british_to_american_dict.json") as f:
    british_to_american_dict = json.load(f)

# set time range and firm list
time_range = list(range(2013,2023))
firm_list = os.listdir(Path("../../report_data"))

# process each firms report each year
for firm in tqdm(firm_list, total = len(firm_list), desc='Processing Firm'):
    master_data_dict = {}
    master_data_dict[firm] = {}
    for year in tqdm(time_range, total = len(time_range), desc='Processing Year'):
        master_data_dict[firm][year] = {}
        try:
            report_path = Path("../../report_data") / firm / f"{year}.pdf"
            
            # text extraction and remove escape sequences and normalise accent characters
            raw = text_extraction(report_path)
            form_feed_split = split_by_form_feed(raw)
            form_feed_split_list_es_removed = remove_escape_sequences_and_Zs(
                normalise_accents(form_feed_split)
            )

            # record raw data stats
            master_data_dict[firm][year]["PDF_File_Size_MB"] = get_file_size_mb(report_path)
            master_data_dict[firm][year]["PDF_Page_Count"] = len(form_feed_split)-1
            master_data_dict[firm][year]["WC_Raw"] = count_words_in_sentences_list(form_feed_split_list_es_removed)
            master_data_dict[firm][year]["CC_Raw"] = count_characters_in_sentences_list_excluding_spaces(form_feed_split_list_es_removed)

            # further text cleaning and processing
            form_feed_split_list_es_and_noise_removed = remove_corrupted_text(
                clean_to_remain_alphanumeric_and_necessary_punctuations(
                    remove_cids_from_list(
                        remove_roman_numerals_from_list(
                            british_to_american_conversion_to_list(
                                form_feed_split_list_es_removed,
                                british_to_american_dict,
                            )
                        )
                    )
                )
            )

            # initialise 4 different text processing recorder for different tasks
            docs_level_sentence_list = [] # raw text with noise removal only
            docs_level_sentence_list_text_normalised = [] #standard NLP preprocessing (lemmatise, lowercasing, americanise, alpha filter, stopword removal, url+email removal, punctuation removal)
            docs_level_sentence_list_text_and_ent_normalised = [] # standard NLP preprocessing + entity removal
            docs_level_sentence_list_text_normalised_ent_joined = [] # standard NLP preprocessing + ngram entites name joined to single token
            docs_level_removed_ent_list = [] # entities removed

            for page in form_feed_split_list_es_and_noise_removed:
                doc = nlp(page)

                doc_level_sentence_list = []
                doc_level_sentence_list_text_normalised = []
                doc_level_sentence_list_text_and_ent_normalised = []
                doc_level_sentence_list_text_normalised_ent_joined = []
                doc_level_removed_ent_list = []

                for sent in list(doc.sents):
                    sent_level_text_normalised_str = ""
                    sent_level_text_and_ent_normalised_str = ""
                    sent_level_text_normalised_ent_joined_str = ""
                    sent_level_string_removed_ent = ""

                    if is_valid_sentence(sent.text):
                        doc_level_sentence_list.append(remove_extra_spaces(remove_starting_punctuation(sent.text.strip())))
                        token_with_ent_iob = []
                        for token in sent:
                            if (
                                not token.like_url
                                and not token.like_email
                                and not token.is_punct
                                and token.is_alpha
                            ):
                                if not token.is_stop:
                                    sent_level_text_normalised_str += token.lemma_.lower() + " "
                                    if token.ent_type_ not in ent_labels_to_remove:
                                        sent_level_text_and_ent_normalised_str += token.lemma_.lower() + " "
                                    else:
                                        sent_level_string_removed_ent += token.lemma_.lower() + " "

                                if token.ent_type_ not in ent_labels_not_to_concate:
                                    if not token.is_stop:
                                        token_with_ent_iob.append((token.text.lower(), str(token.ent_iob_)))
                                    elif token.ent_iob_ == "I":
                                        # this keeps the middle stop word inside entity name such as the "of" in "Bank of England"
                                        token_with_ent_iob.append((token.text.lower(), str(token.ent_iob_)))

                        temp_ent_concate_holder = ""
                        previous_ent_iobs = ""
                        for i, token in enumerate(token_with_ent_iob):
                            sent_length = len(token_with_ent_iob) - 1
                            if token[1] == "B":
                                if previous_ent_iobs == "" or previous_ent_iobs == "O":
                                    temp_ent_concate_holder += token[0]
                                elif previous_ent_iobs == "B" or previous_ent_iobs == "I":
                                    sent_level_text_normalised_ent_joined_str += temp_ent_concate_holder + " "
                                    temp_ent_concate_holder = ""
                                    temp_ent_concate_holder += token[0]
                                previous_ent_iobs = "B"
                            if token[1] == "I":
                                if previous_ent_iobs == "" or previous_ent_iobs == "O":
                                    temp_ent_concate_holder += token[0]
                                elif previous_ent_iobs == "B" or previous_ent_iobs == "I":
                                    temp_ent_concate_holder += "_" + token[0]
                                previous_ent_iobs = "I"
                                if i == sent_length:  # ensure last I token is added
                                    sent_level_text_normalised_ent_joined_str += temp_ent_concate_holder
                            if token[1] == "O":
                                if previous_ent_iobs == "I" or previous_ent_iobs == "B":
                                    sent_level_text_normalised_ent_joined_str += temp_ent_concate_holder + " "
                                    temp_ent_concate_holder = ""
                                    sent_level_text_normalised_ent_joined_str += token[0] + " "
                                else:
                                    sent_level_text_normalised_ent_joined_str += token[0] + " "
                                previous_ent_iobs = "O"
                                
                        doc_level_sentence_list_text_normalised.append(sent_level_text_normalised_str.strip())
                        doc_level_sentence_list_text_and_ent_normalised.append(sent_level_text_and_ent_normalised_str.strip())
                        doc_level_sentence_list_text_normalised_ent_joined.append(sent_level_text_normalised_ent_joined_str.strip())
                        doc_level_removed_ent_list.append(sent_level_string_removed_ent.strip())

                docs_level_sentence_list.append(doc_level_sentence_list)
                docs_level_sentence_list_text_normalised.append(doc_level_sentence_list_text_normalised)
                docs_level_sentence_list_text_and_ent_normalised.append(doc_level_sentence_list_text_and_ent_normalised)
                docs_level_sentence_list_text_normalised_ent_joined.append(doc_level_sentence_list_text_normalised_ent_joined)
                docs_level_removed_ent_list.append(doc_level_removed_ent_list)
                del doc

            master_data_dict[firm][year]["docs_level_sentence_list"] = docs_level_sentence_list
            master_data_dict[firm][year]["docs_level_sentence_list_text_normalised"] = docs_level_sentence_list_text_normalised
            master_data_dict[firm][year]["docs_level_sentence_list_text_and_ent_normalised"] = docs_level_sentence_list_text_and_ent_normalised
            master_data_dict[firm][year]["docs_level_sentence_list_text_normalised_ent_joined"] = docs_level_sentence_list_text_normalised_ent_joined
            master_data_dict[firm][year]["docs_level_removed_ent_list"] = docs_level_removed_ent_list
            
            master_data_dict[firm][year]["WC_docs_level_sentence_list"] = count_words_in_sentences_list([sentences for doc in docs_level_sentence_list for sentences in doc])
            master_data_dict[firm][year]["WC_docs_level_sentence_list_text_normalised"] = count_words_in_sentences_list([sentences for doc in docs_level_sentence_list_text_normalised for sentences in doc])
            master_data_dict[firm][year]["WC_docs_level_sentence_list_text_and_ent_normalised"] = count_words_in_sentences_list([sentences for doc in docs_level_sentence_list_text_and_ent_normalised for sentences in doc])
            master_data_dict[firm][year]["WC_docs_level_sentence_list_text_normalised_ent_joined"] = count_words_in_sentences_list([sentences for doc in docs_level_sentence_list_text_normalised_ent_joined for sentences in doc])
            
        except FileNotFoundError:
            master_data_dict[firm][year] = ["NA"]
            
    master_data_dict_json_data = json.dumps(master_data_dict, indent=4)
    write_path = Path("../../preprocessed_and_parsed_report_data") / f"{firm}_data.json"
    with open(write_path, "w") as file:
        file.write(master_data_dict_json_data)
    del master_data_dict
    del master_data_dict_json_data