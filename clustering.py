import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import seaborn as sns


# %%
def cluster_data(corpus, labels, max_features=500, max_clusters=16, use_idf=True):
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
    v_metrics: Metrics for each number of clusters
    """
    # Instantiate a vectoriser
    tfidf = TfidfVectorizer(max_features=max_features, use_idf=use_idf, lowercase=False, tokenizer=lambda x: x)
    # Fit the vectoriser to the data
    matrix = tfidf.fit_transform(corpus)

    kms = dict()  # A list to store models
    v_metrics = dict()  # A dictionary to store metrics

    print("Clustering the data using {0} features".format(max_features))
    print("Feature names", tfidf.get_feature_names())
    for K in range(2, max_clusters + 1):
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

    return kms, matrix, v_metrics


# %%
def plot_km_model(kms, K, matrix):
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
# Configuration
_max_features = 32
_feature = "triples"
_max_clusters = 30
# Read the preprocessed file into DataFrame
df = pd.read_json("data/preprocessed_data.json")


# Create the corpus based on the tokens in features
# This is data sensitive so it needs to be modified when the feature changes
# The block below is to get the triples data
def process_triples(triples):
    new_triples = []
    for triple in triples:
        st_triples = ' '.join(str(item) for item in triple)
        new_triples.append(st_triples)
    return new_triples


triples = df['triples'].apply(process_triples)
corpus = triples
labels = list(df["category number"])

# %%
# Cluster the data
kms, matrix, v_metrics = cluster_data(corpus, labels, max_features=_max_features, max_clusters=_max_clusters)

# %%
# Plot the clusters
plot_km_model(kms, 16, matrix)


# %%
# Examine the metrics
def examine_metrics(v_metrics):
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
    for K in range(2, max_clusters + 1):
        # For each number of cluster, get the corresponding metric values
        num_of_clusters.append(K)
        silhouettes.append(v_metrics[K]["silhouette"])
        homogeneities.append(v_metrics[K]["homogeneity"])
        completenesses.append(v_metrics[K]["completeness"])
        v_measures.append(v_metrics[K]["v_measure"])
        rand_indices.append(v_metrics[K]["rand_index"])

    # Start plotting
    plt.figure(figsize=[12, 8])
    plotdf = pd.DataFrame()
    plotdf.index = num_of_clusters
    plotdf["Silhouette"] = silhouettes
    plotdf["Homogeneity"] = homogeneities
    plotdf["Completeness"] = completenesses
    plotdf["V Measure"] = v_measures
    plotdf["ARI"] = rand_indices
    plotdf.plot.line()
    plt.title("Metrics for clustering with the number of clusters ranging from 2 to {0}".format(max_clusters))
    plt.show()


# %%
examine_metrics(v_metrics)


# %%
def examine_inertia(_v_metrics):
    plt.figure(figsize=[12, 8])
    plotdf = pd.DataFrame()
    plotdf.index = [i for i in range(2, np.amax(list(_v_metrics.keys())) + 1)]
    plotdf["Inertia"] = [v_metrics[k]["inertia"] for k in range(2, np.amax(list(_v_metrics.keys())) + 1)]
    plotdf.plot.line()
    plt.title("Inertia values corresponding to the number of clusters")
    plt.show()


# %%
examine_inertia(v_metrics)

# %%
