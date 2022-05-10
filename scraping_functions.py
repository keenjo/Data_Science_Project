import numpy as np
import pandas as pd
import pycurl
import re
import spacy
import time
import urllib
import wikipedia
import wptools

from pprint import pprint
from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm

# Defintion of all parameters needed to run scraping functions

# Number of articles you would like to collect per category
_num_results = 100

# Minimum number of sentences you want per article
_min_sents = 5

# Categories for which we want to retrieve elements that are instances of that category
general_categories = {
    "Airports"                 : "wd:Q1248784",
    "Buildings"                : "wd:Q41176",
    "Astronomical objects"     : "wd:Q6999",
    "Cities"                   : "wd:Q7930989",
    "Comics characters"        : "wd:Q1114461",
    "Companies"                : "wd:Q783794",
    "Foods"                    : "wd:Q2095",
    "Transport"                : "wd:Q334166",
    "Sports teams"             : "wd:Q12973014",
    "Language"                 : "wd:Q34770"
}

# Categories for which we want to retrieve elements that have that category as their occupation
occupation_categories = {
    "Artists"                  : "wd:Q483501",
    "Astronauts"               : "wd:Q11631",
    "Politicians"              : "wd:Q82955",
    "Sportspeople"             : "wd:Q50995749",
}

# Categories that contain several subcategories, and for which we want to retrieve elements that are instances of any of those subcategories
multi_categories = [
    "Monuments and memorials",
    "Universities and colleges"
]

# All categories
categories = sorted(list(general_categories.keys()) + list(occupation_categories.keys()) + multi_categories)

sparql = SPARQLWrapper('https://query.wikidata.org/sparql')
sparql.setReturnFormat(JSON)

def get_sparql_request(category):
    '''
    Generates a SPARQL query for retrieving article titles from a given category.
    Arguments:
        `category`: the name of the category
    Returns:
        A SPARQL query for retrieving titles of articles belonging to `category`
    '''
    
    # Add SPARQL prefixes and select all Wikidata items
    query = "PREFIX wd: <http://www.wikidata.org/entity/>\nPREFIX wdt: <http://www.wikidata.org/prop/direct/>\n"
    query += "SELECT ?articlename ?wditem WHERE {{"
    
    if category in occupation_categories.keys():
        # Wikidata elements that have an occupation that is either the category, or of an element that is a subclass of the category
        wikidata_item = occupation_categories[category]
        query += "{{ ?wditem wdt:P106 " + wikidata_item + "}} UNION {{ ?wditem wdt:P106 ?y . ?y wdt:P279* " + wikidata_item + ". }}"
        
    elif category in multi_categories:
        # Wikimedia elements that are an instance of an element that is a subclass of an element that belongs to the Wikimedia Commons category
        query += "?wditem wdt:P31 ?y . ?y wdt:P279* ?z. ?z wdt:P373 \"" + category + "\" ."
        
    elif category in general_categories.keys():
        # Wikidata elements that are either an instance of the category, or of an element that is a subclass of the category
        wikidata_item = general_categories[category]
        query += "{{ ?wditem wdt:P31 " + wikidata_item + "}} UNION {{ ?wditem wdt:P31 ?y . ?y wdt:P279* " + wikidata_item + ". }}"
        
    else:
        raise ValueError("The selected category is of unknown type. Please add it to the correct dictionary.")
    
    # Ensure that the Wikidata page has a corresponding Wikipedia article
    query += " ?link schema:about ?wditem ; schema:isPartOf <https://en.wikipedia.org/> ; schema:name ?articlename . }} LIMIT {0}"
    return query

#%%

def get_articles(cat_list, num_results):
    '''
    Generates a list of lists of Wikipedia article titles (one list per category)
    Arguments:
        `cat_list`:    A list of plain text categories
        `num_results`: The maximum number of articles to retrieve per category
    Returns a tuple containing:
        A list of lists of Wikipedia article titles (one list per category)
        A list of lists of corresponding Wikidata item names (one list per category)
    '''
    
    articles = [] # List of lists of articles (one list per category)
    wd_items = [] # List of lists Wikidata items (one list per category)
    
    for cat in tqdm(cat_list, desc='Retrieving all articles'):
        # For each category, retrieve the list of relevant Wikipedia articles
        sparql_request = get_sparql_request(cat).format(num_results)
        sparql.setQuery(sparql_request)
        # Retry again and again when there's an error
        while True:
            try:
                results = sparql.queryAndConvert()['results']['bindings']
                break
            except urllib.error.HTTPError:
                print('HTTPError caught, trying again in 60 seconds')
                time.sleep(60)
                continue
        
        cat_articles = []
        cat_wd_items = []
        
        # Retrieve individual Wikipedia articles
        for result in tqdm(results, desc=f'Retrieving articles for category [{cat}]'):
            article = result['articlename']['value']
            wd_item = result['wditem']['value']
            
            cat_articles.append(article)
            cat_wd_items.append(wd_item)
        articles.append(cat_articles)
        wd_items.append(cat_wd_items)
        
    return articles, wd_items
        
#%%

def get_titles_info(articles):
    '''
    Retrieves all of the titles and infoboxes for a list of lists of articles (one list per category)
    Arguments:
        `articles`: A list of lists of articles, such as the first list returned by `get_articles`
    Returns a tuple containing two lists:
        - A list of lists of article titles (one list per category)
        - A list of lists of article infoboxes (one list per category)
    '''
    
    titles = []
    infoboxes = []
    
    # For every category
    for i, cat in enumerate(articles):
        cat_titles = []
        cat_boxes = []
        
        # For each article in the given category
        for art in tqdm(cat, desc=f'Retrieving titles & infoboxes ({i + 1}/{len(articles)})'):
            
            # Retrieve the article title
            page = wptools.page(art, silent=True)
            
            # Retry again and again when there's an error
            while True:
                try:
                    page.get_parse()
                    break
                except pycurl.error:
                    print('pycurl error caught, trying again in 60 seconds')
                    time.sleep(60)
                    continue
                    
            page_name = page.data['title']
            cat_titles.append(page_name)
            
            # Retrieve the article infobox
            if page.data['infobox']:
                cat_boxes.append(page.data['infobox'])
            else:
                cat_boxes.append(None)
            
        titles.append(cat_titles)
        infoboxes.append(cat_boxes)
        
    return titles, infoboxes


#%%

def get_content(titles, min_sents):
    '''
    Fetches the content of all of articles whose title is in a given list of lists (one list per category)
    Arguments:
        `titles`:    A list of lists of article titles (one list per category)
        `min_sents`: The minimum number of sentences per article
    Returns:
        A list of lists of content for all input articles (one list per category)
    '''    

    content = []

    # For each category
    for i, cat in enumerate(titles):
        cat_content = []
        
        # For every article title of the category
        for title in tqdm(cat, desc=f'Retrieving content ({i + 1}/{len(titles)})'):
            page_content = ''
            try:
                # Retrieve the corresponding article and its content
                page = wikipedia.page(title, auto_suggest=False, redirect=False)
                page_content = page.content
                
            # In case of error, try using wptools
            except wikipedia.exceptions.PageError:
                try:
                    page = wptools.page(title, silent=True)
                    page.get_query()
                    page_content = page.data['extext']
                except ValueError:
                    pass
            
            # Disambiguation pages are not relevant for us as we would need to pick which page is the correct one
            # Similarly, redirection pages usually redirect to "List of [...]" pages which could be retrieved multiple times
            except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.RedirectError):
                pass
            
            # Check the amount of sentences in the text
            nb_sents = len(list(nlp(page_content).sents))
                    
            if nb_sents > min_sents:
                cat_content.append(page_content)
            else:
                cat_content.append(None)
        content.append(cat_content)
        
    return content

     
#%%

def get_descriptions(wd_items):
    '''
    Fetches the description of all Wikidata items whose name is in a given list of lists (one list per category)
    Arguments:
        `wd_items`: A list of lists of Wikidata item names, such as the second list returned by `get_articles`
    Returns:
        A list of lists of descriptions for all input items (one list per category)
    '''    

    descriptions = []
    
    query = "PREFIX wd: <http://www.wikidata.org/entity/>\nPREFIX schema: <http://schema.org/>"
    query += "SELECT ?desc WHERE {{ <{0}> schema:description ?desc. FILTER ( lang(?desc) = \"en\" ) }}"

    # For each category
    for i, cat in enumerate(wd_items):
        cat_description = []
        
        # For every article title of the category
        for wd_item in tqdm(cat, desc=f'Retrieving description ({i + 1}/{len(wd_items)})'):
            
            # Retrieve the Wikidata description
            sparql.setQuery(query.format(wd_item))
            
            description = ''
            # Retry again and again when there's an error
            while True:
                try:
                    results = sparql.queryAndConvert()['results']['bindings']
                    break
                except urllib.error.HTTPError:
                    print('HTTPError caught, trying again in 60 seconds')
                    time.sleep(60)
                    continue
                
            if results:
                description = results[0]['desc']['value']
            
            if description:
                cat_description.append(description)
            else:
                cat_description.append(None)
        descriptions.append(cat_description)
        
    return descriptions

     
#%%

def get_triples(page_name):
    '''
    Gets all of the triples from the Wikidata page associated with a given Wikipedia article and organizes them in a list
    Arguments:
        `page_name`: The name of the Wikipedia article
    Returns:
        A list of triples from the Wikidata page associated with the article
    '''
    
    # TODO (Max): Maybe this whole function could be converted into a SPARQL query?
    # That way we wouldn't need to split things manually
    # But it would require filtering triples based on whether they appear on Wikidata or not
    
    triples = []
    
    # Retrieve Wikipedia article and associated Wikidata information
    page = wptools.page(page_name, silent=True)
    # Retry again and again when there's an error
    while True:
        try:
            page.get_wikidata()
            break
        except pycurl.error:
            print('pycurl error caught, trying again in 60 seconds')
            time.sleep(60)
            continue
        except LookupError:
            return []
    wd = page.data['wikidata']
    
    # Loop through all triples
    for pred, values in wd.items():
    
        # Write string values directly
        if type(values) is str:
            triples.append([pred, values])
        
        # For lists, add as many triples as there are elements
        if type(values) is list:
            for value in values:
                # The list itself may contain strings or dictionaries
                
                # Write string values directly
                if type(value) is str:
                    triples.append([pred, value])
                    
                # For dicts, add one triple per key-value pair
                elif type(value) is dict:
                    for key, subvalue in value.items():
                        triples.append([pred, key, subvalue])
        
        # For dicts, add one triple per key-value pair
        elif type(values) is dict:
            for subkey, subvalues in values.items():
                # The dictionary values themselves may contain strings or lists
                
                # Write string values directly
                if type(subvalues) is str:
                    triples.append([pred, subkey, subvalues])
                    
                # For lists, add as many triples as there are elements
                elif type(subvalues) is list:
                    for subvalue in subvalues:
                        triples.append([pred, subkey, subvalue])
                
    return triples

#%%

def combine_triples(titles):
    '''
    Gets all of the triples from many Wikidata pages associated with given Wikipedia articles and organizes them in a list of lists
    Arguments:
        `titles`: A list of lists of Wikipedia article titles (one list per category)
    Returns:
        A list of lists of triples from the Wikidata pages associated with the articles
        (one list per category, containing itself one list of triples per article)
    '''
    triples = []
    
    # For each category
    for i, cat in enumerate(titles):
        cat_triples = []
        
        # For each article title in the category
        for art in tqdm(cat, desc=f'Retrieving triples ({i + 1}/{len(titles)})'):
        
            # Retrieve triples and add them to our list
            triples_list = get_triples(art)
            if not triples_list:
                cat_triples.append(None)
            else:
                cat_triples.append(triples_list)
            
        triples.append(cat_triples)
        
    return triples

#%%

def combine_data(categories, titles, infoboxes, content, descriptions, triples):
    
    '''
    Combines all of the data we have collected into a list of lists
    and converts it into a pandas dataframe
    Arguments:
    `categories`:   The list of all of the wikipedia article categories
    `titles`:       The first list returned by `get_titles_info`
    `infoboxes`:    The second list returned by `get_titles_info`
    `content`:      The list returned by `get_content`
    `descriptions`: The list returned by `get_descriptions`
    `triples`:      The list returned by `combine_triples`
    '''

    data = []

    # For each category
    for i, cat in enumerate(categories):
    
        # For each article from the category
        for j in range(len(titles[i])):
        
            # Append all data from one article.
            # Keeping the category number will make it easier to evaluate clustering
            data.append([i, cat, titles[i][j], infoboxes[i][j], content[i][j], descriptions[i][j], triples[i][j]])
            
    # Pandas dataframe containing all of the collected data
    df = pd.DataFrame(data, columns=['Category_num', 'Category', 'Title', 'Infobox', 'Content', 'Description', 'Triples']) 
    
    null_data = {} # The number of null values for each column
    null_data['Title'] = f'{df["Title"].isnull().sum()} null title items'
    null_data['Infobox'] = f'{df["Infobox"].isnull().sum()} null infobox items'
    null_data['Content'] = f'{df["Content"].isnull().sum()} null content items'
    null_data['Description'] = f'{df["Description"].isnull().sum()} null description items'
    null_data['Triples'] = f'{df["Triples"].isnull().sum()} null triples items'
    
    group_data = df.groupby(['Category']).count() # The number of datapoints for each category
    
    return data, group_data, null_data, df

#%%
# Testing

nlp = spacy.load('en_core_web_sm')
articles, wd_items = get_articles(categories, num_results=_num_results)
titles, infoboxes = get_titles_info(articles)
content = get_content(titles, min_sents=_min_sents)
descriptions = get_descriptions(wd_items)
triples = combine_triples(titles)
data, group_data, null_data, df = combine_data(categories, titles, infoboxes, content, descriptions, triples)

#%%

print('>>> Dataframe info <<<')
print(df.info())
print('>>> Grouped data <<<')
print(group_data)
print('>>> Null data <<<')
print(null_data)

#%%

print('Displaying random category data')
index = np.random.randint(0, len(df['Category']))

for column in ['Category', 'Title', 'Infobox', 'Content', 'Description', 'Triples']:
    print('> Column [' + column + ']:')
    pprint(df[column][index])

df.to_json('scraped_data.json')