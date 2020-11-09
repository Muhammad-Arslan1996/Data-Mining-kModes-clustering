import pandas as pd
import numpy as np
from random import choice
import math

def hamming_distance_and_choose_cluster(row, cluster_modes):
    distances = np.array([])
    for cluster_mode in cluster_modes:
        distances = np.append(distances, row.eq(cluster_mode).value_counts().get(False, 0))
    return np.argmin(distances)

#calculate cluster modes after cluster assignment
def calculate_cluster_modes(data, number_of_clusters):
    cluster_modes = []
    for i in range(number_of_clusters):
        mode = data[data['cluster'] == i].mode()
        mode.drop('cluster', 1, inplace=True, errors='ignore')
        cluster_modes.append(mode.loc[0])
    return cluster_modes

#for termination condition
def cluster_sets_are_equal(cluster_modes, new_cluster_modes):
    if(len(cluster_modes) != len(new_cluster_modes)):
        return False
    for i in range(len(cluster_modes)):
        if not cluster_modes[i].equals(new_cluster_modes[i]):
            return False
    return True 


def k_mode_clustering(data, number_of_clusters):
    cluster_centroids = []
    already_chosen_clusters = []
    clusters = {}
    df_with_cluster = data.copy()
    df_with_cluster.drop('p', 1, inplace=True)
    
    #for random centroids, remove the rows with missing values
    data_without_missing_values = df_with_cluster[df_with_cluster['e.1'] != '?'].reset_index()
    
    #chose random centroids
    rows = len(data_without_missing_values)
    for i in range(number_of_clusters):
        rand_index = choice([i for i in range(rows) if i not in already_chosen_clusters])
        already_chosen_clusters.append(rand_index)
        cluster_centroids.append(data_without_missing_values.loc[rand_index])
    new_cluster_centroids = []
    
    iteration = 0
    while not cluster_sets_are_equal(cluster_centroids, new_cluster_centroids):
        print('On iteration number: {}'.format(iteration))
        new_cluster_centroids = cluster_centroids
        df_with_cluster.drop('cluster', 1, inplace=True, errors='ignore')
        df_with_cluster['cluster'] = df_with_cluster.apply(hamming_distance_and_choose_cluster, args=[new_cluster_centroids], axis=1)
        cluster_centroids = calculate_cluster_modes(df_with_cluster, number_of_clusters)
        
        if(iteration == 0):
            for i, acc in enumerate(already_chosen_clusters):
                mask = (df_with_cluster['e.1'] == '?') & (df_with_cluster['cluster'] == i)
                df_with_cluster.loc[mask, 'e.1'] = df_with_cluster.loc[acc]['e.1']
            
        print(len(df_with_cluster[df_with_cluster['e.1'] == '?']))
        iteration += 1

    df_with_cluster.to_csv('prediction.csv')
    return df_with_cluster


def main():
    data = pd.read_csv('./agaricus-lepiota.data')
    result = k_mode_clustering(data, 8)
    print(result['e.1'].unique())


if __name__ == "__main__":
    main()