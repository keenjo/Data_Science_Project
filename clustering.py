import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import seaborn as sns

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

# Lables to evaluate the clustering
labels = list(df["category number"])

# Number of tfidf features that you want to use when testing clustering
_max_features = 32

# Features that you would like to use in the tfidf vectorizer
# Can choose from any of the keys in 'features_dict' above
corpus = 'triples'

# Minimum number of clusters that you would like to test in the cluster data function
_min_clusters = 2

# Maximum number of clusters that you would like to test in the cluster data function
_max_clusters = 32

# List containing the different number of tfidf features you'd like to test
num_features = [10, 30, 50, 70, 90, 100, 150, 200]


# %%
def cluster_data(corpus, labels, max_features=500, min_clusters=2, max_clusters=16, use_idf=True):
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
    corpus = features_dict[corpus]
    
    # Instantiate a vectoriser
    tfidf = TfidfVectorizer(max_features=max_features, use_idf=use_idf, lowercase=False, tokenizer=lambda x: x)
    # Fit the vectoriser to the data
    matrix = tfidf.fit_transform(corpus)

    kms = dict()  # A list to store models
    v_metrics = dict()  # A dictionary to store metrics
    centroids = dict()

    print("Clustering the data using {0} features".format(max_features))
    
    features = tfidf.get_feature_names_out()
    #print("Feature names", tfidf.get_feature_names())
    for K in range(min_clusters, max_clusters + 1):
        print("Current number of clusters: {0}/{1}".format(K, max_clusters))
        # Instantiate the Kmeans model and fit it to the data
        km = KMeans(n_clusters=K, max_iter=500, n_init=15, verbose=0)
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

    return kms, matrix, v_metrics, centroids, features, max_features
              
# %%
def plot_km_model(kms, matrix, K=16):
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
    plt.title("Clustering plot for {0} clusters".format(K))
    plt.show()
    pass

# %%
# Examine the metrics
def examine_metrics(v_metrics, min_clusters=_min_clusters):
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
    plt.show()
    
#%%

def test_num_features(corpus, labels, num_features, num_clusters=16, use_idf=True):
    
    '''
    Function to iterate through multiple numbers of tfidf features
    to see how many features gives us the best result for the kmeans evaluation
    
    max_clusters: number of clusters to use
    - default is 16, but user is encouraged to enter the value that gets the best result
      from the 'cluster_data' function
    '''
    
    corpus = features_dict[corpus]
    
    total_results = {}
        
        
    for num in num_features:
        
        print(f'Testing {num} tfidf features')
        
        # Instantiate a vectoriser
        tfidf = TfidfVectorizer(max_features=num, use_idf=use_idf, lowercase=False, tokenizer=lambda x: x)
        # Fit the vectoriser to the data
        matrix = tfidf.fit_transform(corpus)
        
        # Instantiate the Kmeans model and fit it to the data
        km = KMeans(n_clusters=num_clusters, max_iter=500, n_init=15, verbose=0)
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

def plot_diff_features(data, num_clusters):
    
    '''
    Function to plot the effect of the number of tfidf features on the clustering evaluation
    '''
    
    ax = sns.lineplot(data=data)
    ax.set(xlabel='num_features', 
       ylabel='measures', 
       title=f'Effect of number of features on evaluation metrics using {num_clusters} clusters')
    plt.show()

# %%
def examine_inertia(_v_metrics, min_clusters=_min_clusters, num_features=_max_features):
    '''
    Function to print out the inertia values for different numbers of clusters
    '''
    
    plotdf = pd.DataFrame()
    plotdf.index = [i for i in range(min_clusters, np.amax(list(_v_metrics.keys())) + 1)]
    plotdf["Inertia"] = [v_metrics[k]["inertia"] for k in range(min_clusters, np.amax(list(_v_metrics.keys())) + 1)]
    plotdf.plot.line()
    plt.title(f"Inertia values corresponding to the number of clusters using {_max_features} features")
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

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
        # Get the top 50 terms from each cluster centroid
        for centroid in centroids[num_clusters][x, :num_features]:
            clust_terms.append(features[centroid]) # features output from cluster_data function
            
        top_terms[x+1] = clust_terms
    
    return top_terms

#%%
'''
The idea with this testing section below is that you will first test for the ideal number of clusters
then you will test for the ideal number of features. 

Then by the end you will know the ideal number of 
clusters and features to use with your data.
'''

#%%

# Cluster the data
kms, matrix, v_metrics, centroids, features = cluster_data(corpus, labels, max_features=_max_features, min_clusters=_min_clusters, max_clusters=_max_clusters)

#%%

# Plot how inertia changes with different numbers of clusters
examine_inertia(v_metrics)

#%%

# Plot how the metrics change with different numbers of clusters
examine_metrics(v_metrics)

#%%

# Plot the clusters for one of the kmeans models tested in the cluster_data function
# The last parameter 'K' indicates the number of clusters for which you'd like to see the plot
# K must be between _min_clusters and _max_clusters defined a the top of the script
plot_km_model(kms, matrix, K=16)

#%%

# Test data with different numbers of tfidf features
df_features, feature_num_clusters = test_num_features(corpus, labels, num_features, num_clusters=16, use_idf=True)

# Get the top terms for each cluster (can be printed below if you'd like)
top_terms = check_cluster_features(centroids, labels, features, num_clusters=16, num_features=10)

#%%

# Plot how the metrics change with different numbers of tfidf features
plot_diff_features(df_features, feature_num_clusters)


