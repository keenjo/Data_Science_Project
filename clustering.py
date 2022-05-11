import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import seaborn as sns
import scipy
import os

#%%

# Read the preprocessed file into DataFrame
df = pd.read_json("data/preprocessed_data.json")

#%%
# Dictionary to show the user options of which features they could use for clustering

features_dict = {
    'text': df['processed text'],
    'description': df['processed description'],
    'nouns': df['nouns'],
    'verbs': df['verbs'],
    'NER tokens': df['NER tokens'],
    'lemmas': df['lemmas'],
    'triples': df['processed triples']
    }

#%%
# Definition of all parameters needed for functions below
'''
There is one parameter that must be defined at the bottom of the script for the following functions:
    - plot_km_model
    - check_cluster_features
    - test_num_features
    
These three functions only look at one number of clusters at a time and the idea of this script is
that you will be able to find the ideal number of clusters for your data via the first few clustering
functions. Then these three functions mentioned above will allow you to see the scatterplot
for your ideal number of clusters and test the idea number of tfidf features to go along with
your ideal number of clusters.
'''

# Lables to evaluate the clustering
labels = list(df["category number"])

# Number of tfidf features that you want to use when testing clustering
_max_features = 32

# Features that you would like to use in the tfidf vectorizer
# Can choose from any of the keys in 'features_dict' above
corpus = 'triples'

# If you want to use multiple features in tfidf you can enter them into a list below
# Ex: ['triples', 'lemmas', 'description']
stacked_corpus = None

if stacked_corpus != None:
    hstack = True
else:
    hstack = False

# Minimum number of clusters that you would like to test in the cluster data function
_min_clusters = 2

# Maximum number of clusters that you would like to test in the cluster data function
_max_clusters = 32

# List containing the different number of tfidf features you'd like to test
num_features = [10, 30, 50, 70, 90, 100, 150, 200, 500]

# Number of top terms you would like to see per cluster
num_top_terms = 10

# Naming folder where graphs will be saved (include slash)
# May want to change the firectory name if running the script multiple times
# so older data does not get overwritten
if hstack == False:
    folder_name = f'clustering_results_{corpus}/'
else:
    folder_name = f'clustering_results_{str(stacked_corpus)}/'

#%%
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

directory = make_directory(folder_name)
    
# %%
def cluster_data(corpus, labels, max_features=500, min_clusters=2, max_clusters=16, use_idf=True, hstack=False):
    """
    The function to cluster the feature
    Args:
        corpus: A list of documents to cluster
        labels: A list of labels
        max_features: Maximum number of features
        max_clusters: Maximum number of clusters
        use_idf: True by default if use IDF, False otherwise.

    Returns:
    kms: KMeans model for each number of cluster
    matrix: The matrix vector of the corpus
    v_metrics: Dictionary containing metrics for each number of clusters
               where the key to each dictionary value is the number of clusters
    centroids: Dictionary containing the centroids for each number of clusters
               where the key to each dictionary value is the number of clusters
    features: A list containing the all of the features chosen by the tfidf vectorizer
    """
    if hstack == False:
        corpus = features_dict[corpus]
        
        # Instantiate a vectoriser
        tfidf = TfidfVectorizer(max_features=max_features, use_idf=use_idf, lowercase=False, tokenizer=lambda x: x)
        # Fit the vectoriser to the data
        matrix = tfidf.fit_transform(corpus)
        
        features = tfidf.get_feature_names_out()
    
    else: # If you have decided to test multiple features at once
        
        corpus = [features_dict[feature] for feature in stacked_corpus]
        
        tfidf_stack = []
        features = []
        
        for feature in corpus:
        
            tfidf = TfidfVectorizer(max_features=round(max_features/len(corpus)), use_idf=use_idf, lowercase=False, tokenizer=lambda x: x)
            part_matrix = tfidf.fit_transform(feature)
            tfidf_stack.append(part_matrix)
            
            part_features = tfidf.get_feature_names_out()
            
            for feature in part_features:
                features.append(feature)
            
        matrix = scipy.sparse.hstack(tfidf_stack)

    kms = dict()  # A list to store models
    v_metrics = dict()  # A dictionary to store metrics
    centroids = dict()

    print("Clustering the data using {0} features".format(max_features))
    
    

    for K in range(min_clusters, max_clusters + 1):
        print("Current number of clusters: {0}/{1}".format(K, max_clusters))
        # Instantiate the Kmeans model and fit it to the data
        km = KMeans(n_clusters=K, max_iter=500, n_init=15, verbose=0, random_state=10)
        km.fit(matrix)
        # Get the results
        kms[K] = km  # Model
        # Save the metrics in a dictionary for future uses
        v_metrics[K] = dict()

        v_metrics[K]["inertia"] = km.inertia_  # Inertia
        v_metrics[K]["silhouette"] = metrics.silhouette_score(matrix, km.labels_)  # Silhouette
        v_metrics[K]["homogeneity"] = metrics.homogeneity_score(labels, km.labels_)  # Homogeneity
        v_metrics[K]["completeness"] = metrics.completeness_score(labels, km.labels_)  # Completeness
        v_metrics[K]["v_measure"] = metrics.v_measure_score(labels, km.labels_)  # V Measure
        v_metrics[K]["rand_index"] = metrics.adjusted_rand_score(labels, km.labels_)  # Adjusted Rand Index
        
        centroids[K] = km.cluster_centers_.argsort()

    return kms, matrix, v_metrics, centroids, features
              
# %%
def plot_km_model(kms, matrix, directory=directory, K=16):
    """
    Plot the results of the selected KMeans model

    Args:
        kms: KMeans model
        K: number of clusters
        matrix: input matrix

    Returns:
        Nothing
    """
    # Get the KM model with K number of clusters
    km = kms[K]
    tsne = TSNE(n_components=2, random_state=0, perplexity=30)
    tsne_obj = tsne.fit_transform(matrix)
    tsne_df = pd.DataFrame({'Cluster': km.labels_,
                            'Xtsne': tsne_obj[:, 0],
                            'Ytsne': tsne_obj[:, 1]
                            })
    # Declare the figure
    plt.figure(figsize=[12, 8])
    # Construct the scatter plot
    sns.scatterplot(x='Xtsne', y='Ytsne',
                    hue='Cluster',
                    palette='colorblind',
                    legend='full',
                    data=tsne_df)
    # Set the plot title and show
    plt.title("Clustering plot for {0} clusters and {1} features".format(K, _max_features))
    
    if hstack == False:
        plt.savefig(directory+f'ClusterScatterplot_{corpus}_{K}.png')
    else:
        plt.savefig(directory+f'ClusterScatterplot_stack_{K}.png')
    pass

# %%
# Examine the metrics
def examine_metrics(v_metrics, min_clusters=_min_clusters, directory=directory):
    """
    This takes the following metrics from the clustering and plot them:
        Silhouette scores
        Homogeneity scores
        Completeness scores
        V-Measure scores
        Adjusted Rand indices
    Args:
        v_metrics: the metrics obtained

    Returns:
        Nothing
    """
    # Lists to store metric values
    num_of_clusters = []
    silhouettes = []
    homogeneities = []
    completenesses = []
    v_measures = []
    rand_indices = []
    max_clusters = np.amax(list(v_metrics.keys()))
    for K in range(min_clusters, max_clusters + 1):
        # For each number of cluster, get the corresponding metric values
        num_of_clusters.append(K)
        silhouettes.append(v_metrics[K]["silhouette"])
        homogeneities.append(v_metrics[K]["homogeneity"])
        completenesses.append(v_metrics[K]["completeness"])
        v_measures.append(v_metrics[K]["v_measure"])
        rand_indices.append(v_metrics[K]["rand_index"])

    # Start plotting
    
    # Declare the figure
    plt.figure(figsize=[12, 8])
    
    plotdf = pd.DataFrame()
    plotdf.index = num_of_clusters
    plotdf["Silhouette"] = silhouettes
    plotdf["Homogeneity"] = homogeneities
    plotdf["Completeness"] = completenesses
    plotdf["V Measure"] = v_measures
    plotdf["ARI"] = rand_indices
    plotdf.plot.line()
    plt.title("Metrics for clustering with the number of clusters ranging from {0} to {1}".format(min_clusters, max_clusters))
    plt.xlabel('Number of clusters')
    plt.ylabel('Measures')
    
    if hstack == False:
        plt.savefig(directory+f'ClusterMetrics_{corpus}.png')
    else:
        plt.savefig(directory+f'ClusterMetrics_stack.png')
        
#%%

def test_num_features(corpus, labels, num_features, num_clusters=16, use_idf=True, hstack=False):
    
    '''
    Function to iterate through multiple numbers of tfidf features
    to see how many features gives us the best result for the kmeans evaluation
    
    max_clusters: number of clusters to use
    - default is 16, but user is encouraged to enter the value that gets the best result
      from the 'cluster_data' function
    '''
        
    total_results = {}
        
    if hstack == False:
        corpus = features_dict[corpus]
    else:
        corpus = [features_dict[feature] for feature in stacked_corpus]
        
    for num in num_features:
        
        print(f'Testing {num} tfidf features on {num_clusters} clusters')
        
        if hstack == False:
            # Instantiate a vectoriser
            tfidf = TfidfVectorizer(max_features=num, use_idf=use_idf, lowercase=False, tokenizer=lambda x: x)
            # Fit the vectoriser to the data
            matrix = tfidf.fit_transform(corpus)
    
        else: # If you have decided to 
            tfidf_stack = []
        
            for feature in corpus:
        
                tfidf = TfidfVectorizer(max_features=num, use_idf=use_idf, lowercase=False, tokenizer=lambda x: x)
                part_matrix = tfidf.fit_transform(feature)
                tfidf_stack.append(part_matrix)
            
            matrix = scipy.sparse.hstack(tfidf_stack)
            
        # Instantiate a vectoriser
        #tfidf = TfidfVectorizer(max_features=num, use_idf=use_idf, lowercase=False, tokenizer=lambda x: x)
        # Fit the vectoriser to the data
        #matrix = tfidf.fit_transform(corpus)
        
        # Instantiate the Kmeans model and fit it to the data
        km = KMeans(n_clusters=num_clusters, max_iter=500, n_init=15, verbose=0, random_state=10)
        km.fit(matrix)
        
        # Get the results
        kms[num] = km  # Model
        
        result_values = {}
        
        result_values['Homogeneity'] = metrics.homogeneity_score(labels, km.labels_)
        result_values['Completeness'] = metrics.completeness_score(labels, km.labels_)
        result_values['V measure'] = metrics.v_measure_score(labels, km.labels_)
        result_values['Adjusted Rand Score'] = metrics.adjusted_rand_score(labels, km.labels_)
        result_values['Silhouette Score'] = metrics.silhouette_score(matrix, km.labels_, sample_size=1000)
        
        total_results[num] = result_values
    
    df_features = pd.DataFrame.from_dict(total_results, orient='index')

    return df_features, num_clusters

#%%

def plot_diff_features(data, num_clusters ,directory=directory):
    
    '''
    Function to plot the effect of the number of tfidf features on the clustering evaluation
    '''
    
    # Declare the figure
    plt.figure(figsize=[12, 8])
    
    ax = sns.lineplot(data=data)
    ax.set(xlabel='num_features', 
       ylabel='measures', 
       title=f'Effect of number of features on evaluation metrics using {num_clusters} clusters')
    
    if hstack==False:
        plt.savefig(directory+f'FeaturesTesting_{corpus}_{num_clusters}.png')
    else:
        plt.savefig(directory+f'FeaturesTesting_stack_{num_clusters}.png')

# %%
def examine_inertia(_v_metrics, min_clusters=_min_clusters, num_features=_max_features, directory=directory):
    '''
    Function to print out the inertia values for different numbers of clusters
    '''
    
    # Declare the figure
    plt.figure(figsize=[12, 8])
    
    plotdf = pd.DataFrame()
    plotdf.index = [i for i in range(min_clusters, np.amax(list(_v_metrics.keys())) + 1)]
    plotdf["Inertia"] = [v_metrics[k]["inertia"] for k in range(min_clusters, np.amax(list(_v_metrics.keys())) + 1)]
    plotdf.plot.line()
    plt.title(f"Inertia values corresponding to the number of clusters using {_max_features} features")
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    
    if hstack == False:
        plt.savefig(directory+f'Inertia_{corpus}.png')
    else:
        plt.savefig(directory+'Inertia_stack.png')

# %%

def check_cluster_features(centroids, labels, features, num_clusters=2, num_features=10):
    
    '''
    Function to print out the top features for each cluster
    
    num_clusters = number of clusters for which you would like to see the top terms
    num_features = the number of features you would like to see per cluster
    '''
    
    # Number of cluster labels
    true_k = len(np.unique(labels))
    
    top_terms = {}
    
    # For each item in the labels
    for x in range(true_k):
        clust_terms = []
        # Get the top x terms from each cluster centroid
        for centroid in centroids[num_clusters][x, :num_features]:
            clust_terms.append(features[centroid]) # features output from cluster_data function
            
        top_terms[x+1] = clust_terms
    
    if hstack == False:
        with open(directory+f'TopTerms_{corpus}_{num_clusters}.txt', 'w') as f:
            f.write(f'Top terms for {num_clusters} clusters\n\n')
        
            for key in top_terms:
                f.write(f'Cluster {key}: {top_terms[key]}')
                f.write('\n\n')
                
    else:
        with open(directory+f'TopTerms_stack_{num_clusters}.txt', 'w') as f:
            f.write(f'Top terms for {num_clusters} clusters\n\n')
        
            for key in top_terms:
                f.write(f'Cluster {key}: {top_terms[key]}')
                f.write('\n\n')
                
#%%
'''
The idea with this testing section below is that you will first test for the ideal number of clusters
then you will test for the ideal number of features. 

Then by the end you will know the ideal number of 
clusters and features to use with your data.

All graphs/files created here will be saved in a folder called clustering_results in the same directory that this code is being run from:
    - Inertia graph: Inerta_[name of feature used].png
    - Graph of clustering metrics: ClusterMetrics_[name of feature used].png
    - Scatterplot of clusters: ClusterScatterplot_[name of feature used]_[number of clusters].png
    - Graph of metrics using different numbers of features: FeaturesTesting_[name of feature used]_[number of clusters].png
    - Top terms: TopTerms_[name of features used]_[number of clusters].txt
'''
#%%

# Make directory to store graphs
make_directory(folder_name)

#%%

# Cluster the data
kms, matrix, v_metrics, centroids, features = cluster_data(corpus, labels, max_features=_max_features, min_clusters=_min_clusters, max_clusters=_max_clusters, hstack=hstack)

#%%

# Plot how inertia changes with different numbers of clusters
examine_inertia(v_metrics)

#%%

# Plot how the metrics change with different numbers of clusters
examine_metrics(v_metrics)

#%%

# Plot the clusters for one of the kmeans models tested in the cluster_data function
# The last parameter 'K' indicates the number of clusters for which you'd like to see the plot
# ** K must be between _min_clusters and _max_clusters defined a the top of the script **
plot_km_model(kms, matrix, K=16)

#%%

# Get the top terms for each cluster
# num_clusters is set to 16 since it performed best for our dataset, but you can choose any value between _min_clusters and _max_clusters
check_cluster_features(centroids, labels, features, num_clusters=16, num_features=num_top_terms)

#%%

# Test data with different numbers of tfidf features
# num_clusters is set to 16 since it performed best for our dataset, but you can choose whatever value you want
df_features, feature_num_clusters = test_num_features(corpus, labels, num_features, num_clusters=16, use_idf=True, hstack=hstack)

#%%

# Plot how the metrics change with different numbers of tfidf features
plot_diff_features(df_features, feature_num_clusters)

print('------------------------------------------------------------')
print(f'Clustering and evaluation finished. Results can be found in {directory}\n')