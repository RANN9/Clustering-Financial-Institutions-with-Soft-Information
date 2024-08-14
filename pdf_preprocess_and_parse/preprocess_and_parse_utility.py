# utility functions for preprocessing and parsing pdf files
import os
import re
from unidecode import unidecode
from pdfminer.high_level import extract_text

def get_file_size_mb(pdf_path) -> float:
    """ 
    Get the file size of a PDF in MB.
    """
    file_size = round(os.path.getsize(pdf_path) / 1024 / 1024, 4)
    return file_size

def count_words_in_sentences_list(sentences_list) -> int:
    """
    Count the number of words in a list of sentences split by spaces.
    """
    total_words = sum(len(sentence.split()) for sentence in sentences_list)
    return total_words

def count_characters_in_sentences_list_excluding_spaces(sentences_list) -> int:
    """
    Count the number of characters in a list of sentences excluding spaces.
    """
    total_characters = sum(len(sentence.replace(" ", "")) for sentence in sentences_list)
    return total_characters

def text_extraction(pdf_path) -> str:
    """
    Extract text from a PDF file using pdfminer.high_level.extract_text
    """
    text = extract_text(pdf_path)
    return text

def split_by_form_feed(text) -> list:
    """
    Split text by form feed character.
    """
    return text.split("\x0c")

def normalise_accents(pages_to_clean) -> list:
    """
    Normalise accent characters in the text using unidecode,
    transliterating them to their closest ASCII representation.
    """
    clearned_pages_list = []
    for page in pages_to_clean:
        clearned_pages_list.append(unidecode(page))
    return clearned_pages_list

def remove_escape_sequences_and_Zs(pages_to_clean) -> list:
    """
    Remove various whitespace characters including new lines, tabs, non-breaking spaces,
    figure spaces, and all Unicode 'Separator Space' (Zs) category characters from text.
    Each occurrence is replaced with a regular space.
    """

    regex_pattern_escape_sequences = re.compile(r'[\s\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000]+')
    regex_pattern_escape_sequences

    es_removed_list = []
    for page in pages_to_clean:
        es_removed_list.append(re.sub(regex_pattern_escape_sequences, ' ', page).strip())

    return es_removed_list

def clean_to_remain_alphanumeric_and_necessary_punctuations(pages_to_clean) -> list:
    """
    Clean pages to remain only alphanumeric characters and necessary punctuations.
    """
    regex_pattern_remain_alphanumeric_and_necessary_punctuations = re.compile(r'[^\w\s,.!?;:()/\'\"@-]+')
    regex_pattern_remove_sequences_of_dots = re.compile(r'\.{2,}') # match two or more dots
    regex_pattern_remove_chars = re.compile(r'-{2,}|_{2,}|\\|\\\'')
    
    clearned_pages_list = []

    for page in pages_to_clean:
        clearned_pages_list.append(
            re.sub(regex_pattern_remove_chars,'',
                re.sub(regex_pattern_remove_sequences_of_dots,'',
                    re.sub(regex_pattern_remain_alphanumeric_and_necessary_punctuations, '', page))))
    return clearned_pages_list

def british_to_american_conversion(text, british_to_american_dict) -> str:
    '''
    British English to American English conversion
    '''
    for british, american in british_to_american_dict.items():
        text = text.replace(british, american)
    return text

def british_to_american_conversion_to_list(pages_to_clean, british_to_american_dict) -> list:
    '''
    British English to American English conversion to list
    '''
    converted_pages_list = []
    for page in pages_to_clean:
        converted_pages_list.append(british_to_american_conversion(page, british_to_american_dict))
    return converted_pages_list

def remove_roman_numerals(sentence_str: str) -> str:
    regex_fullmatch_pattern = re.compile(r"(?i)^(M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})|[a-zA-Z0-9])$")
    # regex pattern to fullmatch roman numerals up to 4000 or any single alphanumeric to remove
    words = sentence_str.split()
    cleaned_words = []
    for word in words:
        if not regex_fullmatch_pattern.fullmatch(word):
            cleaned_words.append(word)
    return " ".join(cleaned_words)

def remove_roman_numerals_from_list(pages_to_clean) -> list:
    """
    Remove roman numerals from a list of sentences.
    """
    clearned_pages_list = []
    for page in pages_to_clean:
        clearned_pages_list.append(remove_roman_numerals(page))
    return clearned_pages_list

def remove_cids(sentence_str: str) -> str:
    # Regex pattern to match cid:xxx and remove them
    regex_cid_pattern = re.compile(r"\bcid:[a-zA-Z0-9]+\b")
    cleaned_sentence = regex_cid_pattern.sub('', sentence_str)
    cleaned_sentence = ' '.join(cleaned_sentence.split())
    return cleaned_sentence

def remove_cids_from_list(pages_to_clean) -> list:
    """
    Remove cids from a list of sentences.
    """
    clearned_pages_list = []
    for page in pages_to_clean:
        clearned_pages_list.append(remove_cids(page))
    return clearned_pages_list

def remove_corrupted_text(pages_to_clean) -> list:
    """
    Remove continous alphanumeric seperated by space (usually alphanumeric resembled logos or caps in pdf file)
    """
    regex_pattern_continous_alphanumeric_seperated_by_space = re.compile(r'(\b[A-Za-z0-9_]\s+){2,}[A-Za-z0-9_]?\b')
    clearned_pages_list = []
    for page in pages_to_clean:
        clearned_pages_list.append(re.sub(regex_pattern_continous_alphanumeric_seperated_by_space, '', page).strip())
    return clearned_pages_list

def is_valid_sentence(sentence_str) -> bool:
    """
    Check if a sentence is valid. (Contains at least one word and that word has at least one alphabet character)
    """
    if len(sentence_str.split()) >= 1 and any(char.isalpha() for char in sentence_str):
        return True
    else:
        return False
    
def remove_starting_punctuation(sentence) -> str:
    """
    Remove starting punctuation from a sentence.
    """
    regex_pattern_starting_colons = re.compile(r'^[,.!?;:]+|^[,.!?;:]+\s+')
    return re.sub(regex_pattern_starting_colons,'',sentence)

def remove_extra_spaces(sentence) -> str:
    """
    Remove extra spaces from a sentence.
    """
    regex_pattern_extra_spaces = re.compile(r'\s+')
    return re.sub(regex_pattern_extra_spaces, ' ', sentence).strip()
