import wptools
import wikipedia

#%%

def get_articles(cat_list, num_results): # CHANGE THIS FUNCTION SO WE DON'T GET SO MANY 'LIST ARTICLES'
    
    '''
    This function takes a list of categories and gets a specified number
    of wikipedia articles relating to that category:
        - cat_list: list of categories
        - num_results: number of articles to get for each category
    '''
    
    
    articles_total = [] # List containing a list of articles for each category
    
    for cat in cat_list:
        new_articles = wikipedia.search(f'{cat}', results=num_results)
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
    
    for cat in article_list: # For every category in article_list
        cat_titles = []
        cat_boxes = []
        for art in cat: # For every article in a specific category
            page = wptools.page(art)
            page.get_parse()
            page_name = page.data['title']
            cat_titles.append(page_name)
            titles_total.append(cat_titles)
            
            if page.data['infobox']:
                page_infobox = page.data['infobox']
                cat_boxes.append(page_infobox)
            else:
                cat_boxes.append(None)
                
        titles_total.append(cat_titles)
        infoboxes_total.append(cat_boxes)
        
    return titles_total, infoboxes_total
    
    
#%%

def get_content(title_list): # MODIFY THIS FUNCTION OR 'get_articles' TO CHECK FOR NUMBER OF SENTENCES PER ARTICLE

    '''
    This function gets the content from all of the article titles in the total_titles list
    - title_list: list of all of the article titles retreived thus far
    '''    

    content_total = [] # content of the pages

    for cat in title_list: # For every category in title_list
        cat_cont_list = [] # List of all of the content for a specific category (i.e. category content list)
        for title in cat: # For every title in a specific category
            try:
                page = wikipedia.page(title)
                cat_cont_list.append(page.content)
            except wikipedia.exceptions.PageError:
                title_list.remove(title)
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
    page.get_wikidata()
    
    for x in page.data['wikidata']:
        one_triple = []
        if type(page.data['wikidata'][x]) is str:
            one_triple.append(x)
            one_triple.append(page.data["wikidata"][x])
            triples_list.append(one_triple)
            one_triple = []
        if type(page.data['wikidata'][x]) is list:
            for item in page.data['wikidata'][x]:
                if type(item) is str:
                    one_triple.append(x)
                    one_triple.append(item)
                    triples_list.append(one_triple)
                    one_triple = []
                elif type(item) is dict:
                    for key in item:
                        one_triple.append(x)
                        one_triple.append(key)
                        one_triple.append(item[key])
                        triples_list.append(one_triple)
                        one_triple = []
        if type(page.data['wikidata'][x]) is dict:
            for item in page.data['wikidata'][x]:
                one_triple.append(x)
                one_triple.append(item)
                one_triple.append(page.data["wikidata"][x][item])
                triples_list.append(one_triple)
                one_triple = []
                
    return triples_list

#%%

def combine_triples(titles_total):
    
    triples_total = [] # All of the triples for every articles
    
    for cat in titles_total: # For each category (inner list) within titles_total
        cat_triples = [] # All of the triples for one category
        for art in cat: # For every article in a specific category
            triples_list = get_triples(art)
            cat_triples.append(triples_list)
            
        triples_total.append(cat_triples)
        
    return triples_total

#%%

def combine_data(categories, titles, infoboxes, content, triples):
    
    '''
    This function combines all of the data we have collected into one dictionary dictionaries
    whcih could easily be converted into a pandas dataframe later
    '''

    data = {}

    data['category'] = categories
    data['title'] = titles
    data['infobox'] = infoboxes
    data['content'] = content
    data['triples'] = triples

    return data

