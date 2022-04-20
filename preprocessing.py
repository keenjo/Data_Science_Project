from spacy.lang.en.stop_words import STOP_WORDS
from tqdm import tqdm

import pandas as pd
import spacy
import string

nlp = spacy.load('en_core_web_sm')

# Add tqdm support to pandas by creating the `DataFrame.progress_apply()` method
tqdm.pandas(desc='Pre-processing article content and descriptions')

# Note. To improve clustering and classification results, feel free to
# add further pre-processing steps (eg Named entity recognition, pos-
# tagging and extraction of e.g., nouns and verbs); or/and to also pre-
# process the wikidata statements and the infobox content (note that
# these are not standard text however)

def preprocess(text):
    '''
    Processes a text by tokenizing it and removing stop words and punctuation from the output.
    Arguments:
        `text` The input text string.
    Returns a list of tokens of `text`, without punctuation or stopwords.
    '''
    
    if text is None:
        return []
    
    tokens = []
    # Tokenize
    
    sp_text = nlp(text)
    
    # Remove stopwords and punctuation
    for sent in sp_text.sents:
        for token in sent:
            if token.text.lower() not in STOP_WORDS and not all(char in string.punctuation for char in token.text):
                tokens.append(token)
    return tokens

df = pd.read_json('scraped_data.json')

# Convert to a new dataframe
data = zip(
    df['Category_num'],
    df['Category'],
    df['Title'],
    df['Content'],
    df['Content'].progress_apply(preprocess),
    df['Description'],
    df['Description'].progress_apply(preprocess)
)

converted_df = pd.DataFrame(data, columns=['Category number', 'Category', 'Title', 'Text', 'Processed text', 'Description', 'Processed description'])
converted_df = converted_df.dropna()
converted_df = converted_df.reset_index(drop=True)
converted_df.to_json('preprocessed_data.json', default_handler=str)
#converted_df.to_json('preprocessed_data.json', default_handler=str) 
