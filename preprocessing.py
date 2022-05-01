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
            if token.text.lower() not in STOP_WORDS and not all(char in string.punctuation for char in token.text) and not token.is_space and not token.is_digit:
                tokens.append(token)
    return tokens


def pos(tokens):
    '''
    Function to extract POS
    
    Parameters
    ----------
    tokens: a list of tokens

    Returns
    -------
    pos: a list of lists in which the format of each inner list is [POS, token]

    '''
    
    pos = []

    for token in tokens:
        pos_val = []
        pos_val.append(token.pos_)
        pos_val.append(token)
        pos.append(pos_val)
        
    return pos


def nouns(tokens):
    # Function to only extract noun POS tokens from text
    pos = []
    
    for token in tokens:
        if token[0] == 'NOUN':
            pos.append(token[1])
            
    return pos


def verbs(tokens):
    # Function to only extract verb POS tokens from text
    
    pos = []
    
    for token in tokens:
        if token[0] == 'VERB':
            pos.append(token[1])
            
    return pos
  
        
def ner(text):
    '''
    Function for Named Entity Recognition
    
    Parameters
    ----------
    text: a string of spacy processed text
    
    Returns
    -------
    ner: a list of lists in which the format of each inner list is [NE, token]
    '''
    
    sp_text = nlp(text)
    
    ner = []
    
    for word in sp_text.ents:
        ner_val = []
        ner_val.append(word.label_)
        ner_val.append(word)
        ner.append(ner_val)
        
    return ner


def ner_tokens(tokens):
    # Function to only extract the NER tokens without their labels
    ner = []
    
    for word in tokens:
        ner.append(word[1])
        
    return ner


def lemma(tokens):
    '''
    Function to extract lemmas from the list of tokens

    Parameters
    ----------
    tokens: a list of tokens

    Returns
    -------
    lemmas: a list of lemmatized tokens

    '''
    lemmas = []
    
    for token in tokens:
        lemmas.append(token.lemma_)
        
    return lemmas


def drop_null(df, drop_description=False):
    '''
    Function to drop any rows which have null text, descriptions, or triples
    
    Parameters
    ----------
    df : dataframe
    drop_description: parameter to choose whether or not you want to drop rows that do not have a description

    Returns
    -------
    df : processed dataframe

    '''
        
    for row, value in enumerate(df['Content']):
        
        # If the 'Content' column is empty, drop the corresponding row
        if value is None:
            df.drop(row, axis=0, inplace=True)
            
        # If the 'Triples' column is empty, drop the corresponding row
        elif df['Triples'][row] is None:
            df.drop(row, axis=0, inplace=True)
            
        # If the 'Description' column is empty, drop the corresponding row [Only if 'drop_description' variable is True]
        if drop_description == True:
            if df['Description'][row] is None:
                df.drop(row, axis=0, inplace=True)
            
    # Reset the indexes after dropping all of the values
    df = df.reset_index(drop=True)
            
    return df


def process_triples(triples):
    
    new_triples = []
   
    for triple in triples:
        st_triples = ' '.join(str(item) for item in triple)
        new_triples.append(st_triples)
            
    return new_triples


df = pd.read_json('scraped_data.json')

df = drop_null(df, drop_description=False)

preprocessed_text = df['Content'].progress_apply(preprocess)
POS = preprocessed_text.progress_apply(pos)
NER = df['Content'].progress_apply(ner)

# Convert to a new dataframe
data = zip(
    df['Category_num'],
    df['Category'],
    df['Title'],
    df['Content'],
    preprocessed_text,
    df['Description'],
    df['Description'].progress_apply(preprocess),
    POS,
    POS.progress_apply(nouns),
    POS.progress_apply(verbs),
    NER,
    NER.progress_apply(ner_tokens),
    preprocessed_text.progress_apply(lemma),
    df['Triples'],
    df['Triples'].progress_apply(process_triples)
)

converted_df = pd.DataFrame(data, columns=['category number', 'category', 'title', 'text', 'processed text', 'description', 'processed description', 
                                           'POS', 'nouns', 'verbs', 'NER', 'NER tokens', 'lemmas', 'triples', 'processed triples'])
converted_df.to_json('preprocessed_data.json', default_handler=str)
 
group_data = converted_df.groupby(['category']).count() # The number of datapoints for each category

print('>>> Grouped data <<<')
print(group_data)
