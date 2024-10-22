import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import binom
from collections import defaultdict, Counter
from itertools import combinations
import umap

def sample_80_percent(df):
    """Sample 80% of the rows and columns of a dataframe."""
    sampled_rows = np.random.choice(df.index, size=int(df.shape[0] * 0.8), replace=False)
    sampled_cols = np.random.choice(df.columns, size=int(df.shape[1] * 0.8), replace=False)
    return df.loc[sampled_rows, sampled_cols]

def cluster_labels(df, method='ward', metric='euclidean', n_clusters=3):
    """Cluster dataframe and return cluster labels."""
    return list(zip(df.index, fcluster(linkage(df, method=method, metric=metric), n_clusters, criterion='maxclust')))

def cluster_sizes(cluster_labels):
    """Return the size of each cluster in a list of cluster labels."""
    return Counter(list(cluster_labels))

def get_pairs(sample_list):
    """Return all pairs of samples in a list."""
    unsorted_pairs = combinations(sample_list, 2)
    sorted_pairs = [tuple(sorted(x)) for x in unsorted_pairs]
    return sorted_pairs

def cluster_sample_dict(cluster_labels):
    """Return a dictionary with cluster labels as keys and lists of samples in that cluster as values."""
    cluster_sample_dict = defaultdict(list)
    for sample, cluster in cluster_labels:
        cluster_sample_dict[cluster].append(sample)
    return cluster_sample_dict

def pca_transform(df, n_components):
    """Perform PCA on a dataframe and return the transformed dataframe."""
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(df.T)
    return pd.DataFrame(transformed_data, index=df.columns)

def umap_transform(df, n_components):
    """Perform UMAP on a dataframe and return the transformed dataframe."""
    reducer = umap.UMAP(n_components=n_components)
    transformed_data = reducer.fit_transform(df.T)
    return pd.DataFrame(transformed_data, index=df.columns)


## Bootstrapping tools:  

def generate_bootstraps(df, n_bootstraps=50):
    """Generate a list of bootstrapped samples using a generator."""
    for _ in range(n_bootstraps):
        yield sample_80_percent(df)

def all_sampled_samples(bootstraps):
    """Return a set of all samples in a list of bootstraps."""
    all_sampled_samples = set()
    for bootstrap in bootstraps:
        all_sampled_samples.update(bootstrap.T.index)
    return all_sampled_samples

def cluster_bootstraps_pca(df, n_components, n_bootstraps, method='ward', metric='euclidean', n_clusters=3):
    """Perform bootstrapped clustering on a list of bootstrapped samples and return the cluster labels for each bootstrap."""
    cluster_labels_list = []
    for bootstrap in generate_bootstraps(df, n_bootstraps):
        bootstrap_pca = pca_transform(bootstrap, n_components)
        cluster_labels_list.append(cluster_labels(bootstrap_pca, method=method, metric=metric, n_clusters=n_clusters))
    return cluster_labels_list

def kmeans_bootstraps_pca(df, n_components, n_bootstraps, n_clusters=3):
    """Perform bootstrapped kmeans clustering on a list of bootstrapped samples and return the cluster labels for each bootstrap."""
    cluster_labels_list = []
    for bootstrap in generate_bootstraps(df, n_bootstraps):
        bootstrap_pca = pca_transform(bootstrap, n_components)
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(bootstrap_pca)
        cluster_labels_list.append(list(zip(bootstrap_pca.index, kmeans.labels_)))
    return cluster_labels_list

def cluster_bootstraps_umap(df, n_components, n_bootstraps, method='ward', metric='euclidean', n_clusters=3):
    """Perform bootstrapped clustering on a list of bootstrapped samples and return the cluster labels for each bootstrap."""
    cluster_labels_list = []
    for bootstrap in generate_bootstraps(df, n_bootstraps):
        bootstrap_umap = umap_transform(bootstrap, n_components)
        cluster_labels_list.append(cluster_labels(bootstrap_umap, method=method, metric=metric, n_clusters=n_clusters))
    return cluster_labels_list

def kmeans_bootstraps_umap(df, n_components, n_bootstraps, n_clusters=3):
    """Perform bootstrapped kmeans clustering on a list of bootstrapped samples and return the cluster labels for each bootstrap."""
    cluster_labels_list = []
    for bootstrap in generate_bootstraps(df, n_bootstraps):
        bootstrap_umap = umap_transform(bootstrap, n_components)
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(bootstrap_umap)
        cluster_labels_list.append(list(zip(bootstrap_umap.index, kmeans.labels_)))
    return cluster_labels_list

def within_cluster_pairs(cluster_sample_dict):
    """Return all pairs of samples within each cluster."""
    within_cluster_pairs = []
    for samples in cluster_sample_dict.values():
        within_cluster_pairs.extend(get_pairs(samples))
    return within_cluster_pairs

def random_cluster_pairs(cluster_labels, sample_list):
    """Return clusters of equal size to main clusters, but with random pairs of samples."""
    all_pairs = get_pairs(sample_list)
    sizes = [len(v) for v in cluster_sample_dict(cluster_labels).values()]
    pair_cluster_sizes = [(x*(x-1)/2) for x in sizes]

    # Create a 1-dimensional array of indices to select random pairs
    indices = np.arange(len(all_pairs))
    random_indices = np.random.choice(indices, size=int(sum(pair_cluster_sizes)), replace=False)
    random_pairs = [all_pairs[idx] for idx in random_indices]

    return random_pairs

def stability_score(result_list_of_lists):
    """Calculate the rate at which pairs of samples cluster together across bootstraps."""
    # Create set of all pairs in result_list_of_lists
    all_pairs = set()
    for result_list in result_list_of_lists:
        all_pairs.update(result_list)
    
    # Loop over all pairs and count the number of times each pair appears in result_list_of_lists
    pair_scores = []
    pairs_and_scores = {}
    for pair in all_pairs:
        count = 0
        for result_list in result_list_of_lists:
            if pair in result_list:
                count += 1
        score = count/len(result_list_of_lists)
        pair_scores.append(score)
        pairs_and_scores[pair] = score
    
    return pair_scores, pairs_and_scores

def average_stability_score_per_sample(pairs_and_scores):
    """Calculate the average stability score for each sample."""
    sample_scores = defaultdict(list)
    for pair, score in pairs_and_scores.items():
        sample_scores[pair[0]].append(score)
        sample_scores[pair[1]].append(score)
    return {k: np.mean(v) for k, v in sample_scores.items()}


## Main functions:

def cluster_stability_pca(df, n_components, method='ward', metric='euclidean', n_clusters=3, n_bootstraps=50):
    """Perform dimensionality reduction and bootstrapped clustering and determine the rate at which pairs of samples cluster together."""
    
    # Perform bootstrapped dim red and clustering on random samples of the data
    cluster_labels_boots = cluster_bootstraps_pca(df, n_components=n_components, n_bootstraps=n_bootstraps, method=method, metric=metric, n_clusters=n_clusters)

    # Establish clustered pairs and random pairs per bootstrap
    sampled_samples = all_sampled_samples(generate_bootstraps(df, n_bootstraps))

    cluster_pairs = [within_cluster_pairs(cluster_sample_dict(cluster_labels)) for cluster_labels in cluster_labels_boots]

    random_pairs = [random_cluster_pairs(cluster_labels, sampled_samples) for cluster_labels in cluster_labels_boots]

    # Calculate stability scores
    cluster_pair_stability_scores, pairs_and_scores_dict = stability_score(cluster_pairs)

    random_pair_stability_scores, _ = stability_score(random_pairs)

    per_sample_stability_scores = average_stability_score_per_sample(pairs_and_scores_dict)

    mean_cluster_stability_score = np.mean(cluster_pair_stability_scores)

    mean_random_stability_score = np.mean(random_pair_stability_scores)

    num_successes = int(mean_cluster_stability_score * len(cluster_pairs))

    total_trials = len(cluster_pairs) + len(random_pairs)

    # Calculate p-value
    p_value = binom.sf(num_successes, total_trials, mean_random_stability_score)

    return mean_cluster_stability_score, mean_random_stability_score, p_value, cluster_pair_stability_scores, random_pair_stability_scores, per_sample_stability_scores


def kmeans_cluster_stability_pca(df, n_components, n_clusters = 3):

    """perform dimensionality reduction and bootstrapped kmeans clustering and determine the rate at which pairs of samples cluster together"""

    bootstraps = generate_bootstraps(df)

    # perform bootstrapped dim red and clustering on random samples of the data
    cluster_labels_boots = kmeans_bootstraps_pca(bootstraps, n_clusters = n_clusters, n_components = n_components)

    # establish clustered pairs and random pairs per hootstrap
    sampled_samples = all_sampled_samples(bootstraps)

    cluster_pairs = [within_cluster_pairs(cluster_sample_dict(cluster_labels)) for cluster_labels in cluster_labels_boots]

    random_pairs = [random_cluster_pairs(cluster_labels, sampled_samples) for cluster_labels in cluster_labels_boots]

    # calculate stability scores

    cluster_pair_stability_scores, pairs_and_scores_dict = stability_score(cluster_pairs)

    random_pair_stability_scores, _ = stability_score(random_pairs)

    per_sample_stability_scores = average_stability_score_per_sample(pairs_and_scores_dict)

    mean_cluster_stability_score = np.mean(cluster_pair_stability_scores)

    mean_random_stability_score = np.mean(random_pair_stability_scores)

    num_successes = int(mean_cluster_stability_score * len(cluster_pairs))

    total_trials = len(cluster_pairs) + len(random_pairs)

    # calculate p-value
    p_value = binom.sf(num_successes, total_trials, mean_random_stability_score)

    return mean_cluster_stability_score, mean_random_stability_score, p_value, cluster_pair_stability_scores, random_pair_stability_scores, per_sample_stability_scores

def cluster_stability_umap(df, n_components, method='ward', metric='euclidean', n_clusters=3, n_bootstraps=50):
    """Perform dimensionality reduction and bootstrapped clustering and determine the rate at which pairs of samples cluster together."""
    
    # Perform bootstrapped dim red and clustering on random samples of the data
    cluster_labels_boots = cluster_bootstraps_umap(df, n_components=n_components, n_bootstraps=n_bootstraps, method=method, metric=metric, n_clusters=n_clusters)

    # Establish clustered pairs and random pairs per bootstrap
    sampled_samples = all_sampled_samples(generate_bootstraps(df, n_bootstraps))

    cluster_pairs = [within_cluster_pairs(cluster_sample_dict(cluster_labels)) for cluster_labels in cluster_labels_boots]

    random_pairs = [random_cluster_pairs(cluster_labels, sampled_samples) for cluster_labels in cluster_labels_boots]

    # Calculate stability scores
    cluster_pair_stability_scores, pairs_and_scores_dict = stability_score(cluster_pairs)

    random_pair_stability_scores, _ = stability_score(random_pairs)

    per_sample_stability_scores = average_stability_score_per_sample(pairs_and_scores_dict)

    mean_cluster_stability_score = np.mean(cluster_pair_stability_scores)

    mean_random_stability_score = np.mean(random_pair_stability_scores)

    num_successes = int(mean_cluster_stability_score * len(cluster_pairs))

    total_trials = len(cluster_pairs) + len(random_pairs)

    # Calculate p-value
    p_value = binom.sf(num_successes, total_trials, mean_random_stability_score)

    return mean_cluster_stability_score, mean_random_stability_score, p_value, cluster_pair_stability_scores, random_pair_stability_scores, per_sample_stability_scores

