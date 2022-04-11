import wptools
import wikipedia
import pandas as pd
import numpy as np
import re
import spacy

from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm

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
    "Written communication"    : "wd:Q1149626"
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

categories = sorted(list(general_categories.keys()) + list(occupation_categories.keys()) + multi_categories)

def get_sparql_request(category):
    '''
    Generates a SPARQL that returns a list of titles of Wikipedia articles about items of the given category.
        - category: the name of the category
    '''
    
    # Add SPARQL prefixes and select all Wikidata items
    query = "PREFIX wd: <http://www.wikidata.org/entity/>\nPREFIX wdt: <http://www.wikidata.org/prop/direct/>\n"
    query += "SELECT ?articlename WHERE {{"
    
    
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

def get_articles(cat_list, num_results): #min_sents):
    # **Add parameter to check number of sentences per article (either here or in 'get_content' function)**
    # Tried creating this function with RDF/SPARQL but had trouble looping through queries for each category,
    # but this method seems to be working well, so it should be fine
    
    '''
    This function takes a list of categories and gets a specified number
    of wikipedia articles relating to each category:
        - cat_list: list of categories
        - num_results: number of articles to get for each category
    '''
    
    sparql = SPARQLWrapper('https://query.wikidata.org/sparql')
    sparql.setReturnFormat(JSON)
    
    #nlp = spacy.load('en_core_web_sm') <- part of testing number of sentences
    
    articles_total = [] # List containing a list of articles for each category
    
    for cat in tqdm(cat_list, desc='Retrieving articles'): # For each category in the 'categories' list
        sparql_request = get_sparql_request(cat).format(num_results)
        sparql.setQuery(sparql_request)
        results = sparql.queryAndConvert()['results']['bindings']
        
        new_articles = []
        for result in results: # For every article that was retrieved for a specific category
            article = result['articlename']['value']
            add_article = True
        
            '''
            * Idea I was testing to check number of sentences (not currently working)
            
            page = wptools.page(article)
            page.get_query()
            text = page.data['extext']
            sp_text = nlp(text)
            
            sents = []
    
            for sentence in sp_text.sents:
                    sents.append(sentence)
            if len(sents) < min_sents:
                add_article = False
            '''
            # I commented these out because they shouldn't be necessary anymore
            # if re.search('lists?', article.lower()): # Get rid of article if 'list' or 'lists' is in the title
                # add_article = False
            # elif re.search('disambiguation', article.lower()): # Get rid of article if 'disambiguation' is in the title
                # add_article = False
                
            if add_article:
                new_articles.append(article)
        articles_total.append(new_articles)
        
    return articles_total
        
#%%

def get_titles_info(article_list):
    
    '''
    This function gets all of the titles & infoboxes for the articles fetched via the get_articles function
    - article_list: list of articles returned from the get_articles function
    '''
    
    titles_total = [] # Contains all of the titles 
    infoboxes_total = [] # infoboxes of the pages
    counter = 0
    
    for cat in article_list: # For every category in article_list
        cat_titles = []
        cat_boxes = []
        counter += 1
        for art in tqdm(cat, desc=f'Retrieving titles & infoboxes ({counter}/{len(article_list)})'): # For every article in a specific category
            page = wptools.page(art)
            page.get_parse()
            page_name = page.data['title']
            cat_titles.append(page_name)
            print(f'{art} title retrieved')
            
            if page.data['infobox']:
                page_infobox = page.data['infobox']
                cat_boxes.append(page_infobox)
                print(f'{art} infobox retrieved')
            else:
                cat_boxes.append(None)
                print(f'{art} does not have an infobox')
            
        titles_total.append(cat_titles)
        infoboxes_total.append(cat_boxes)
        
    return titles_total, infoboxes_total
    
    
#%%

def get_content(titles_total, infoboxes_total):

    '''
    This function gets the content from all of the article titles in the total_titles list
    - titles_total: list of all of the article titles retreived
    - infoboxes_total: list of all of the infoboxes retreived
    
    **I used the wikipedia package rather than wptools here because the wikipedia package seems to get way more text than wptools**
    '''    

    content_total = [] # content of the pages

    for cat_index, cat in enumerate(titles_total): # For every category in title_list
        cat_cont_list = [] # List of all of the content for a specific category
        for title_index, title in tqdm(enumerate(cat), desc=f'Retreiving content ({cat_index + 1}/{len(titles_total)})'): # For every title in a specific category
            try:
                page = wikipedia.page(title)
                cat_cont_list.append(page.content)
                print(f'{title} content retrieved')  
                
                ''' 
                For the exceptions below, I just try to retrieve content using wptools rather than wikipedia
                - Even though wptools retrieves less content than the wikipedia package, I figure it is better than
                  putting a null value for the content of an article; null values are only entered if neither wikipedia
                  nor wptools can find the content
                '''
            except wikipedia.exceptions.PageError:
                try:
                    page = wptools.page(title)
                    page.get_query()
                    page_content = page.data['extext']
                    cat_cont_list.append(page_content)
                    print(f'{title} content retrieved')
                except ValueError:
                    cat_cont_list.append(None)
                    print(f'{title} content not found')
            except wikipedia.exceptions.DisambiguationError: #as many:
                try:
                    page = wptools.page(title)
                    page.get_query()
                    page_content = page.data['extext']
                    cat_cont_list.append(page_content)
                    print(f'{title} content retrieved')
                except ValueError:
                    cat_cont_list.append(None)
                    print(f'{title} content not found')
               
        content_total.append(cat_cont_list)
        
    return content_total

     
#%%

def get_triples(page_name):
    
    '''
    This function gets all of the triples for a given page and
    organizes them in a list of lists
    '''
    
    triples_list = [] # List of all of the triples for one article
    
    page = wptools.page(page_name, silent=True)
    
    try:
        page.get_wikidata()
        
        for x in page.data['wikidata']: # For each triple in the list of wikidata triples
            if type(page.data['wikidata'][x]) is str:
                one_triple = [] # lists for one triple, to be added to the full list of triples 'triples_list'
                one_triple.append(x)
                one_triple.append(page.data["wikidata"][x])
                triples_list.append(one_triple)
            if type(page.data['wikidata'][x]) is list:
                for item in page.data['wikidata'][x]:
                    if type(item) is str:
                        one_triple = []
                        one_triple.append(x)
                        one_triple.append(item)
                        triples_list.append(one_triple)
                    elif type(item) is dict:
                        for key in item:
                            one_triple = []
                            one_triple.append(x)
                            one_triple.append(key)
                            one_triple.append(item[key])
                            triples_list.append(one_triple)
            elif type(page.data['wikidata'][x]) is dict:
                for key in page.data['wikidata'][x]:
                    value = page.data["wikidata"][x][key]
                    if type(value) is str:
                        one_triple = []
                        one_triple.append(x)
                        one_triple.append(key)
                        one_triple.append(value)
                        triples_list.append(one_triple)
                    elif type(value) is list:
                        for index in value:
                            one_triple = []
                            one_triple.append(x)
                            one_triple.append(key)
                            one_triple.append(index)
                    
    except LookupError:
        pass
                
    return triples_list

#%%

def combine_triples(titles_total):
    
    triples_total = [] # All of the triples for every articles
    counter = 0
    
    for cat in titles_total: # For each category (inner list) within titles_total
        cat_triples = [] # All of the triples for one category
        counter += 1
        for art in tqdm(cat, desc=f'Retrieving triples ({counter}/{len(titles_total)})'): # For each article in a specific category
            triples_list = get_triples(art)
            if not triples_list:
                cat_triples.append(None)
                print(f'{art} triples not found')
            else:
                cat_triples.append(triples_list)
                print(f'{art} triples retrieved')
            
        triples_total.append(cat_triples)
        
    return triples_total

#%%

def combine_data(categories, titles, infoboxes, content, triples):
    
    '''
    This function combines all of the data we have collected into a list of lists
    and converts it into a pandas dataframe
    - categories: list of all of the wikipedia article categories
    - titles: titles_total list from get_titles_info function
    - infoboxes: infoboxes_total from get_titles_info function
    - content: content_total from get_content function
    - triples: triples_total from combine triples function
    '''

    data_total = [] # List of lists containing all data

    for cat_num, cat in enumerate(categories): # For every category in categories list
        for index in range(len(titles[cat_num])): # For each item in each specific category
            single_data = [] # List containing data from one article

            single_data.append(cat_num) # Keeping category number will make it easier to evaluate clustering
            single_data.append(cat)
            single_data.append(titles[cat_num][index])
            single_data.append(infoboxes[cat_num][index])
            single_data.append(content[cat_num][index])
            single_data.append(triples[cat_num][index])
            
            data_total.append(single_data) # Append all data from one article to data_total list
            
    df = pd.DataFrame(data_total, columns = ['Category_num', 'Category', 'Title', 'Infobox', 'Content', 'Triples']) # Pandas dataframe containing all of the collected data
    
    null_data = {} # The number of null values for each column
    null_data['Title'] = f'{df["Title"].isnull().sum()} null title items'
    null_data['Infobox'] = f'{df["Infobox"].isnull().sum()} null infobox items'
    null_data['Content'] = f'{df["Content"].isnull().sum()} null content items'
    null_data['Triples'] = f'{df["Triples"].isnull().sum()} null triples items'
    
    group_data = df.groupby(['Category']).count() # The number of datapoints for each category
    
    print('Data successfully combined.')
    
    return data_total, group_data, null_data, df

#%%
# Testing

articles_total = get_articles(categories, num_results=50)
titles_total, infoboxes_total = get_titles_info(articles_total)
content_total = get_content(titles_total, infoboxes_total)
triples_total = combine_triples(titles_total)
data_total, group_data, null_data, df = combine_data(categories, titles_total, infoboxes_total, content_total, triples_total)

#%%

print(df.info())
print(group_data)
print(null_data)

#%%

index = np.random.randint(0,len(df['Category']))
print(f'Category: {df["Category"][index]}\n')
print(f'Title: {df["Title"][index]}\n')
print(f'Infobox: {df["Infobox"][index]}\n')
print(f'Content: {df["Content"][index]}\n')
print(f'Triples: {df["Triples"][index]}\n')



