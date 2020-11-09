import pandas as pd
import numpy as np
from random import choice
import math

def hamming_distance_and_choose_cluster(data, cluster_modes):
    clusters = []
    for idx, cluster_mode in enumerate(cluster_modes):
        df = data.iloc[:, 0:22]
        clusters.append('c' + str(idx))
        data['c' + str(idx)] = df.ne(cluster_mode).sum(1)
    data['cluster'] = (data.loc[:, clusters].idxmin(axis = 1))
    data['cluster'] = data['cluster'].str[1:].astype(int)
    data = data.drop(columns=clusters)
    return data

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
    
    #chose random centroids
    rows = len(df_with_cluster)
    i = 0
    while i < number_of_clusters:
        rand_index = choice([i for i in range(rows) if i not in already_chosen_clusters])
        value = df_with_cluster.loc[rand_index]
        if value['e.1'] == '?':
            continue
        already_chosen_clusters.append(rand_index)
        cluster_centroids.append(df_with_cluster.loc[rand_index])
        i += 1
    new_cluster_centroids = []
    
    iteration = 0
    while not cluster_sets_are_equal(cluster_centroids, new_cluster_centroids):
        print('On iteration number: {}'.format(iteration))
        new_cluster_centroids = cluster_centroids
        df_with_cluster.drop('cluster', 1, inplace=True, errors='ignore')
        #df_with_cluster['cluster'] = df_with_cluster.apply(hamming_distance_and_choose_cluster, args=[new_cluster_centroids], axis=1)
        df_with_cluster = hamming_distance_and_choose_cluster(df_with_cluster, new_cluster_centroids)
        cluster_centroids = calculate_cluster_modes(df_with_cluster, number_of_clusters)
        
        if(iteration == 0):
            for i, acc in enumerate(already_chosen_clusters):
                mask = (df_with_cluster['e.1'] == '?') & (df_with_cluster['cluster'] == i)
                df_with_cluster.loc[mask, 'e.1'] = df_with_cluster.loc[acc]['e.1']
            
        iteration += 1

    df_with_cluster.to_csv('prediction.csv')
    return df_with_cluster


def main():
    data = pd.read_csv('./agaricus-lepiota.data')
    result = k_mode_clustering(data, 13)


if __name__ == "__main__":
    main()