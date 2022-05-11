from spacy.lang.en.stop_words import STOP_WORDS
from tqdm import tqdm

import pandas as pd
import spacy
import string
import os

nlp = spacy.load('en_core_web_sm')

# Define the directory where your scraped data is stored so it can be loaded into a dataframe
df = pd.read_json('data/scraped_data.json')

# Add tqdm support to pandas by creating the `DataFrame.progress_apply()` method
tqdm.pandas(desc='Pre-processing article content and descriptions')

def make_directory(folder_name):
    '''
    Function to create a directory for the results graphs
    Parameters
    ----------
    folder_name: name of a folder as a string (defined at the beginning of the script)
    Returns
    -------
    directory: a directory where graphs will be stored
    '''
    
    try:
        directory = folder_name
        os.mkdir(directory)
    except FileExistsError:
        pass
    
    return directory


def preprocess(text):
    '''
    Processes a text by tokenizing it and removing stop words and punctuation from the output.
    
    Parameters
    ----------
    text: the input text string

    Returns
    -------
    tokens: a list of tokens from `text`, without punctuation or stopwords
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


def pos(new_tokens):
    '''
    Function to extract POS
    
    Parameters
    ----------
    new_tokens: a list of tokens

    Returns
    -------
    pos: a list of lists in which the format of each inner list is [POS, token]

    '''
    
    pos = []

    for token in new_tokens:
        pos_val = []
        pos_val.append(token.pos_)
        pos_val.append(token)
        pos.append(pos_val)
        
    return pos


def nouns(pos_tokens1):
    '''
    Function to extract noun POS tokens from text
    
    Parameters
    ----------
    pos_tokens1: a list of tokens

    Returns
    -------
    noun_list: a list of nouns
    '''

    noun_list = []
    
    for token in pos_tokens1:
        if token[0] == 'NOUN':
            noun_list.append(token[1])
            
    return noun_list


def verbs(pos_tokens2):
    '''
    Function to extract verb POS tokens from text
    
    Parameters
    ----------
    pos_tokens2: a list of tokens

    Returns
    -------
    verb_list: a list of verbs
    '''
    
    verb_list = []
    
    for token in pos_tokens2:
        if token[0] == 'VERB':
            verb_list.append(token[1])
            
    return verb_list
  
        
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


def ner_tokens(ner_tokens):
    '''
    Function to only extract NER tokens without their labels

    Parameters
    ----------
    ner_tokens: a list of tokens

    Returns
    -------
    ner: a list of NER tokens

    '''


    ner = []
    
    for word in ner_tokens:
        ner.append(word[1])
        
    return ner


def lemma(raw_tokens):
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
    
    for token in raw_tokens:
        lemmas.append(token.lemma_.lower())
        
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
    df: processed dataframe

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
    '''
    Function to process each triple into a string rather than a list
    
    Parameters
    ----------
    triples : a list of lists in which each inner list is a group of triples

    Returns
    -------
    new_triples : a list of triples where each triple is a string
    '''
    
    new_triples = []
   
    for triple in triples:
        st_triples = ' '.join(str(item) for item in triple)
        new_triples.append(st_triples.lower())
            
    return new_triples


def lowercase(upper_tokens):
    '''
    Function to lowercase all tokens/text that is not already lowercase
    
    Parameters
    ----------
    upper_tokens : a list of tokens or text

    Returns
    -------
    lower_tokens : same list that was entered into the function but everythin will be lowercased
    '''
    
    lower_tokens = []
    
    for word in upper_tokens:
        lower_tokens.append(str(word).lower())
    
    return lower_tokens


make_directory('data/')

df = drop_null(df, drop_description=False)

preprocessed_text = df['Content'].progress_apply(preprocess)
preprocessed_description = df['Description'].progress_apply(preprocess)
POS = preprocessed_text.progress_apply(pos)
nouns_list = POS.progress_apply(nouns)
verbs_list = POS.progress_apply(verbs)
NER = df['Content'].progress_apply(ner)
ner_tokens_list = NER.progress_apply(ner_tokens)

# Convert to a new dataframe
data = zip(
    df['Category_num'],
    df['Category'],
    df['Title'],
    df['Content'],
    preprocessed_text.progress_apply(lowercase),
    df['Description'],
    preprocessed_description.progress_apply(lowercase),
    POS,
    nouns_list.progress_apply(lowercase),
    verbs_list.progress_apply(lowercase),
    NER,
    ner_tokens_list.progress_apply(lowercase),
    preprocessed_text.progress_apply(lemma),
    df['Triples'],
    df['Triples'].progress_apply(process_triples)
)

converted_df = pd.DataFrame(data, columns=['category number', 'category', 'title', 'text', 'processed text', 'description', 'processed description', 
                                           'POS', 'nouns', 'verbs', 'NER', 'NER tokens', 'lemmas', 'triples', 'processed triples'])
converted_df.to_json('data/preprocessed_data.json', default_handler=str)
 
group_data = converted_df.groupby(['category']).count() # The number of datapoints for each category

print('>>> Grouped data <<<')
print(group_data)
