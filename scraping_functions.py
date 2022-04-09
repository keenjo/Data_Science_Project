import wptools
import wikipedia
import pandas as pd
import numpy as np
import re
from tqdm import tqdm

#%%
#I've changed the names of some of these categories from plural to singular because the API gives much better results
# For instance, with 'Airports' I get a lot of list articles but with 'Airport' I get articles about actual airports

categories = ['Airport', 'Artist', 'Astronaut', 'Building', 'Astronomical Object',
              'City', 'Comicbook Character', 'Company', 'Food', 'Transport', 'Monument',
              'Politician', 'Team', 'Athlete', 'University', 'Written Communication']


#%%

def get_articles(cat_list, num_results):
    # **Add parameter to check number of sentences per article (either here or in 'get_content' function)
    # Tried creating this function with RDF/SPARQL but had trouble looping through queries for each category
    
    '''
    This function takes a list of categories and gets a specified number
    of wikipedia articles relating to that category:
        - cat_list: list of categories
        - num_results: number of articles to get for each category
    '''
    
    
    articles_total = [] # List containing a list of articles for each category
    
    for cat in tqdm(cat_list, desc='Retrieving articles'):
        new_articles = wikipedia.search(f'{cat}', results=num_results)
        for article in new_articles:
            if re.search('list', article.lower()):
                new_articles.remove(article)
            elif re.search('lists', article.lower()):
                new_articles.remove(article)
            elif re.search('disambiguation', article.lower()):
                new_articles.remove(article)
                
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
    - title_list: list of all of the article titles retreived
    - infobox_list: list of all of the infoboxes retreived
    '''    

    content_total = [] # content of the pages

    for cat_index, cat in enumerate(titles_total): # For every category in title_list
        cat_cont_list = [] # List of all of the content for a specific category
        for title_index, title in tqdm(enumerate(cat), desc=f'Retreiving content ({cat_index + 1}/{len(titles_total)})'): # For every title in a specific category
            try:
                page = wikipedia.page(title)
                cat_cont_list.append(page.content)
                print(f'{title} content retrieved')
            except wikipedia.exceptions.PageError:
                # ***Don't love this solution but don't know if there is an alternative to solve PageError***
                cat_cont_list.append(None)
                print(f'{title} content not found')
            except wikipedia.exceptions.DisambiguationError: #as many:
                cat_cont_list.append(None)
                print(f'{title} content not found')
                # Had a double disambiguation error which broke the code below, working on solution
                
                '''
                This bit will choose a random value from the disambiguation options then do the following:
                    - update the information in the title and infoboxes lists
                    - retrieve the content for this new article
                    
                NOTE TO SELF: Make this bit more efficient by simplifying functions and making this bit recursive
                - instead of having all of these functions only work on a list of articles,
                  make them work on one article then just create another function which loops through this simpler funtion
                
                updated_item = many.options[np.random.randint(0, len(many.options))]
                
                page = wptools.page(updated_item)
                page.get_parse()
                page_name = page.data['title']
                titles_total[cat_index][title_index] = page_name 
                print(f'UPDATE: {title} title retrieved')
            
                if page.data['infobox']:
                    page_infobox = page.data['infobox']
                    infoboxes_total[cat_index][title_index] = page_infobox
                    print(f'UPDATE: {title} infobox retrieved')
                else:
                    infoboxes_total[cat_index][title_index] = None
                    print(f'UPDATE: {title} does not have an infobox')
                
                page = wikipedia.page(updated_item)
                cat_cont_list.append(page.content)
                
                print(f'{title} content retrieved')
                '''
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
        
        for x in page.data['wikidata']:
            one_triple = []
            #print(type(page.data['wikidata'][x]))
            if type(page.data['wikidata'][x]) is str:
                one_triple.append(x)
                one_triple.append(page.data["wikidata"][x])
                triples_list.append(one_triple)
                one_triple = []
                #print(f'{x}; {page.data["wikidata"][x]}\n')
            if type(page.data['wikidata'][x]) is list:
                for item in page.data['wikidata'][x]:
                    if type(item) is str:
                        one_triple.append(x)
                        one_triple.append(item)
                        triples_list.append(one_triple)
                        one_triple = []
                        #print(f'{x}; {item}\n')
                    elif type(item) is dict:
                        for key in item:
                            one_triple.append(x)
                            one_triple.append(key)
                            one_triple.append(item[key])
                            triples_list.append(one_triple)
                            one_triple = []
                            #print(f'{x}, {key}, {item[key]}\n')
            if type(page.data['wikidata'][x]) is dict:
                for item in page.data['wikidata'][x]:
                    one_triple.append(x)
                    one_triple.append(item)
                    one_triple.append(page.data["wikidata"][x][item])
                    triples_list.append(one_triple)
                    one_triple = []
                    #print(f'{x}; {item}; {page.data["wikidata"][x][item]}\n')
                    
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
        for art in tqdm(cat, desc=f'Retrieving triples ({counter}/{len(titles_total)})'):
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
            
            single_data.append(cat)
            single_data.append(titles[cat_num][index])
            single_data.append(infoboxes[cat_num][index])
            single_data.append(content[cat_num][index])
            single_data.append(triples[cat_num][index])
            
            data_total.append(single_data) # Append all data from one article to data_total list
            
    df = pd.DataFrame(data_total, columns = ['Category', 'Title', 'Infobox', 'Content', 'Triples'])
    
    null_data = {}
    null_data['Title'] = f'{df["Title"].isnull().sum()} null title items'
    null_data['Infobox'] = f'{df["Infobox"].isnull().sum()} null infobox items'
    null_data['Content'] = f'{df["Content"].isnull().sum()} null content items'
    null_data['Triples'] = f'{df["Triples"].isnull().sum()} null triples items'
    
    group_data = df.groupby(['Category']).count()
    
    print('Data successfully combined.')
    
    return data_total, group_data, null_data, df

#%%
# Testing

articles_total = get_articles(categories, num_results=25)
titles_total, infoboxes_total = get_titles_info(articles_total)
content_total = get_content(titles_total, infoboxes_total)
triples_total = combine_triples(titles_total)
data_total, group_data, null_data, df = combine_data(categories, titles_total, infoboxes_total, content_total, triples_total)

#%%

print(df.head())
print(group_data)
print(null_data)
    


