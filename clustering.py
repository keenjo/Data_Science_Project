import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import seaborn as sns


# %%
def cluster_data(corpus, labels, max_features=500, use_idf=True):
    """
    The function to cluster the feature
    Args:

        corpus: A list of documents to cluster
        labels: A list of labels
        max_features: Maximum number of features
        use_idf: True by default if use IDF, False otherwise.

    Returns:
    kms: KMeans model for each number of cluster
    matrix: The matrix vector of the corpus
    v_metrics: Metrics for each number of clusters
    """
    # Instantiate a vectoriser
    tfidf = TfidfVectorizer(max_features=max_features, use_idf=use_idf)
    # Fit the vectoriser to the data
    matrix = tfidf.fit_transform(corpus)

    kms = dict()  # A list to store models
    v_metrics = dict()  # A dictionary to store metrics

    print("Clustering the data using {0} features".format(max_features))

    for K in range(2, 17):
        print("Current number of clusters: {0}/{1}".format(K, 16))
        # Instantiate the Kmeans model and fit it to the data
        km = KMeans(n_clusters=K, init='k-means++', max_iter=300, n_init=5, verbose=0)
        km.fit(matrix)
        # Get the results
        kms[K] = km  # Model
        # Save the metrics in a dictionary for future uses
        v_metrics[K] = dict()

        v_metrics[K]["inertia"] = km.inertia_  # Inertia
        v_metrics[K]["silhouette"] = metrics.silhouette_score(matrix, km.labels_, metric="euclidean")  # Silhouette
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
    tsne = TSNE(n_components=2, random_state=0, perplexity=35)
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
_max_features = 500
_feature = "triples"

# Read the preprocessed file into DataFrame
df = pd.read_json("preprocessed_data.json")

# Create the corpus based on the tokens in features
# This is data sensitive so it needs to be modified when the feature changes
# The block below is to get the triples data
tokens = []
for item in df["triples"]:
    mini_list = [mini_item for mini_item in item]
    mini_tokens = []
    for micro_list in mini_list:
        micro_token = [str(micro_item) for micro_item in micro_list]
        mini_tokens.append(" ".join(micro_token))
    tokens.append(" ".join(mini_tokens))

corpus = tokens
labels = df["category number"]

# Cluster the data
kms, matrix, v_metrics = cluster_data(corpus, labels, max_features=_max_features)

# %%
# Plot the clusters
plot_km_model(kms, 16, matrix)


# %%
# Examine the metrics
def examine_metrics(v_metrics):
    # Lists to store metric values
    num_of_clusters = []
    inertias = []
    silhouettes = []
    homogeneities = []
    completenesses = []
    v_measures = []
    rand_indices = []

    for K in range(2, 17):
        # For each number of cluster, get the corresponding metric values
        num_of_clusters.append(K)
        inertias.append(v_metrics[K]["inertia"])
        silhouettes.append(v_metrics[K]["silhouette"])
        homogeneities.append(v_metrics[K]["homogeneity"])
        completenesses.append(v_metrics[K]["completeness"])
        v_measures.append(v_metrics[K]["v_measure"])
        rand_indices.append(v_metrics[K]["rand_index"])

    plt.figure(figsize=[12, 8])
    plt.xlabel("Clusters")

    plotdf = pd.DataFrame()
    # plotdf["Number of clusters"] = num_of_clusters
    plotdf.index = num_of_clusters
    plotdf["Silhouette"] = silhouettes
    plotdf["Homogeneity"] = homogeneities
    plotdf["Completeness"] = completenesses
    plotdf["V Measure"] = v_measures
    plotdf["ARI"] = rand_indices
    plotdf.plot.line()
    plt.title("Metrics for clustering with the number of clusters ranging from 2 to 16")
    plt.show()


examine_metrics(v_metrics)

# %%
