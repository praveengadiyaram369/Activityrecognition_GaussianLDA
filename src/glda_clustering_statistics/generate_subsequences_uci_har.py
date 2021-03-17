# _importing required libraries
import os
import sys
import collections
import pickle
import json

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from scipy import stats
import statistics


sensory_words_traindf = pd.DataFrame()
sensory_words_testdf = pd.DataFrame()
words_embedding_dict = {}


def load_train_test_data(input_file_path, col_names):

    main_df = pd.read_csv(input_file_path, names=col_names)
    main_df = main_df.astype({'subject_id': int, 'activityID': int})

    return main_df


def subsequence_statistics(subsequences):

    Mean = []
    Standard_deviation = []
    Skewness = []
    IQR = []
    # Min=[]
    # Max=[]
    # Median=[]
    # Range=[]
    Lower_quartile = []
    Middle_quartile = []
    Upper_quartile = []
    # Coefficient_of_variation=[]
    # Kurtosis=[]
    for i in range(0, len(subsequences)):

        mean = sum(subsequences[i])/len(subsequences[i])
        Mean.append(mean)

        std = statistics.stdev(subsequences[i])
        Standard_deviation.append(std)

        # Cov=std/mean
        # Coefficient_of_variation.append(Cov)

        # minimum=min(subsequences[i])
        # Min.append(minimum)

        # maximum=max(subsequences[i])
        # Max.append(maximum)

        # range1=maximum-minimum
        # Range.append(range1)

        skewness = stats.skew(subsequences[i])
        Skewness.append(skewness)

        # median=statistics.median(subsequences[i])
        # Median.append(median)

        q3, q2, q1 = np.percentile(subsequences[i], [75, 50, 25])

        Lower_quartile.append(q1)

        Middle_quartile.append(q2)

        Upper_quartile.append(q3)

        iqr = q3 - q1
        IQR.append(iqr)

        # kurtosis=stats.kurtosis(subsequences[i])
        # Kurtosis.append(kurtosis)

    data = list(zip(Mean, Standard_deviation, Skewness, IQR))
    statistic_feature_df = pd.DataFrame(
        data, columns=['Mean', 'Standard_deviation', 'Skewness', 'IQR'])

    return statistic_feature_df


def window_sampling(main_df, window_length, window_overlap, flag_train=False):

    max_window_index = len(main_df.index)
    sequence_names = main_df.columns.tolist()
    num_of_subsequences = len(sequence_names)
    sub_sequences = [[] for x in range(num_of_subsequences)]

    window_index = 0

    while window_index <= (max_window_index - window_length):

        activity_sequence = main_df[sequence_names[1]
                                    ][window_index:window_index+window_length].tolist()
        subject_sequence = main_df[sequence_names[0]
                                   ][window_index:window_index+window_length].tolist()
        if len(set(activity_sequence)) == 1:
            sub_sequences[1].append(activity_sequence[0])
            sub_sequences[0].append(subject_sequence[0])

            for idx in range(2, num_of_subsequences):
                sub_sequences[idx].append(
                    main_df[sequence_names[idx]][window_index:window_index+window_length].tolist())

        window_index += window_overlap

    # _converting into numpy arrays
    np_sequences = np.asarray(sub_sequences[2:])
    # print(np_sequences.shape)
    # Initializing sensory words dataframe with null values
    #sensory_words_traindf = pd.DataFrame(columns=sequence_names)
    if flag_train:
        sensory_words_traindf['subject_id'] = sub_sequences[0]
        sensory_words_traindf['activityID'] = sub_sequences[1]
    else:
        sensory_words_testdf['subject_id'] = sub_sequences[0]
        sensory_words_testdf['activityID'] = sub_sequences[1]

    statistics_list = []
    for idx in range(0, np_sequences.shape[0]):
        statistic_df_axis = subsequence_statistics(np_sequences[idx])
        statistics_list.append(statistic_df_axis)

    return statistics_list


# assigning words for each cluster
def get_assigned_words(seq_clusters, cluster_words, axis, flag_train=False):

    # _assign word to each cluster of the subsequence usnig numpy where function
    assigned_words = np.where(
        seq_clusters != 0, seq_clusters, cluster_words[0])
    for idx in range(1, len(cluster_words)):
        assigned_words = np.where(
            seq_clusters != idx, assigned_words, cluster_words[idx])

    if flag_train:
        sensory_words_traindf[axis] = assigned_words
        assigned_clusterWord = pd.DataFrame(
            data=assigned_words, columns=['cluster_word'])
        return assigned_clusterWord
    else:
        sensory_words_testdf[axis] = assigned_words


def clustering(statistic_train_df, statistic_test_df, axis, cluster_cnts, cluster_words):

    statistic_train_df = Normalizer().fit_transform(np.array(statistic_train_df))
    statistic_test_df = Normalizer().fit_transform(np.array(statistic_test_df))

    model = KMeans(n_clusters=cluster_cnts).fit(statistic_train_df)

    cluster_ids = pd.DataFrame(model.predict(
        statistic_train_df), columns=['cluster ID'])
    cluster_test_ids = pd.DataFrame(model.predict(
        statistic_test_df), columns=['cluster ID'])

    seq_clusters = cluster_ids.to_numpy()
    assigned_clusterWord = get_assigned_words(
        seq_clusters, cluster_words, axis, flag_train=True)
    get_assigned_words(cluster_test_ids.to_numpy(), cluster_words, axis)

    centroids_of_clusters = pd.DataFrame(model.cluster_centers_[cluster_ids['cluster ID']],
                                         columns=['Mean_c', 'Standard_deviation_c', 'Skewness_c', 'IQR_c'])
    result = pd.concat([assigned_clusterWord, centroids_of_clusters], axis=1)
    result = result.drop_duplicates()

    return result


# generating names for cluster count
def generate_cluster_names(sequence_names, cluster_cnt=100):

    words_dict = {}

    for seq in sequence_names:
        prefix = seq
        words_dict[seq] = [prefix+'_'+str(i) for i in range(cluster_cnt)]

    return words_dict


def cluster_word_sort(axis_clusters, cluster_names):

    result = axis_clusters.loc[(
        axis_clusters['cluster_word'] == cluster_names)]

    return result.iloc[:, 1:]


def perform_clustering(statistics_train, statistics_test, channels, cluster_cnts, words_generation_flag=False):

    clusters_centroid = []
    centroid_statistic = []

    words_dict = generate_cluster_names(channels, cluster_cnts)

    for statistic_train_df, statistic_test_df, axis in zip(statistics_train, statistics_test, channels):

        cluster_names = words_dict[axis]
        axis_clusters = clustering(
            statistic_train_df, statistic_test_df, axis, cluster_cnts, cluster_names)

        clusters_centroid.append(axis_clusters)

        for j in range(len(cluster_names)):

            cluster_stats = cluster_word_sort(axis_clusters, cluster_names[j])
            centroid_statistic.append(cluster_stats)
            words_embedding_dict[cluster_names[j]
                                 ] = cluster_stats.values[0].tolist()

    # stop words generation
    stop_words_generation(channels)
    if words_generation_flag:
        # new words generations inter sensor channels for train
        new_words_generation(channels, flag_train=True)
        # new words generations inter sensor channels for test
        new_words_generation(channels)
        embeddings_filepath = os.getcwd(
        ) + f'/../../data/sub_sequence_output/word_embeddings_from_clusters.json'
        with open(embeddings_filepath, 'w') as fp:
            json.dump(words_embedding_dict, fp)
    else:

        embeddings_filepath = os.getcwd(
        ) + f'/../../data/sub_sequence_output/word_embeddings_from_clusters.txt'
        pd.concat(centroid_statistic).to_csv(
            embeddings_filepath, index=False, header=False)

    # writing train documents to text files
    write_clustering_output(sensory_words_traindf.columns[2:], flag_train=True)
    # writing test documents to text files
    write_clustering_output(sensory_words_testdf.columns[2:])

    print(f'Finished generate_subsequences_uci_har  : {cluster_cnts} ')


def stop_words_generation(channels):

    stop_word_list = []

    def each_channel(channel):

        stopwords = sensory_words_traindf[[channel, 'activityID']].groupby(channel)[
            'activityID'].nunique()
        stopwords = stopwords[stopwords > 3].keys().tolist()

        return stopwords

    for channel in channels:

        stop_word_list.extend(each_channel(channel))

    with open(os.getcwd() + f'/../../data/stopwords.pkl', 'wb') as f:
        pickle.dump(stop_word_list, f)


def write_clustering_output(channels, flag_train=False):

    if flag_train:

        # _combine individual words as documents
        sensory_words_traindf['final_sub_sequence'] = sensory_words_traindf[channels].apply(
            lambda row: ' '.join(row.values.astype(str)), axis=1)
        # _save the combined values to text files
        for subject in sensory_words_traindf['subject_id'].unique():
            activity = sensory_words_traindf.loc[(
                sensory_words_traindf['subject_id'] == subject)]['activityID'].values[0]
            output_filepath = os.getcwd() + f'/../../data/documents/train/activity_subseq_' + \
                str(subject) + '_' + str(activity) + '.txt'
            sensory_words_traindf.loc[(sensory_words_traindf['subject_id'] == subject)][[
                'final_sub_sequence']].to_csv(output_filepath, sep='\t', index=False, header=False)

    else:

        # _combine individual words as documents
        sensory_words_testdf['final_sub_sequence'] = sensory_words_testdf[channels].apply(
            lambda row: ' '.join(row.values.astype(str)), axis=1)
        # _save the combined values to text files
        for subject in sensory_words_testdf['subject_id'].unique():
            activity = sensory_words_testdf.loc[(
                sensory_words_testdf['subject_id'] == subject)]['activityID'].values[0]
            output_filepath = os.getcwd() + f'/../../data/documents/test/activity_subseq_' + \
                str(subject) + '_' + str(activity) + '.txt'
            sensory_words_testdf.loc[(sensory_words_testdf['subject_id'] == subject)][[
                'final_sub_sequence']].to_csv(output_filepath, sep='\t', index=False, header=False)


def form_words(row, flag_train=False):

    temp = []
    temp = row.values
    if flag_train:
        # words_embedding_dict[temp[0]+temp[1]+temp[2]] =  [x + y + z
        # for x, y, z in zip(words_embedding_dict[temp[0]], words_embedding_dict[temp[1]], words_embedding_dict[temp[2]])]

        words_embedding_dict[temp[0]+temp[1]+temp[2]+temp[3]] = [v1+v2+v3+v4
                                                                 for v1, v2, v3, v4 in zip(words_embedding_dict[temp[0]], words_embedding_dict[temp[1]], words_embedding_dict[temp[2]], words_embedding_dict[temp[2]])]

    return ''.join(temp.astype(str))


def new_words_generation(channels, flag_train=False):

    word_combinations = [['X1', 'Y1', 'Y2', 'Z2'], ['X1', 'Z1', 'X2', 'Y2'], ['Y1', 'Z1', 'X2', 'Z2'], [
        'X1', 'Y1', 'X2', 'Z2'], ['X1', 'Z1', 'Y2', 'Z2'], ['Y1', 'Z1', 'X2', 'Y2']]

    for idx, combinations in enumerate(word_combinations):
        acc_axis = combinations[0]
        temp = combinations[1:]
        if flag_train:
            sensory_words_traindf[acc_axis + temp[0] + temp[1] + temp[2]] = sensory_words_traindf[[
                acc_axis, temp[0], temp[1] + temp[2]]].apply(lambda row: form_words(row, flag_train), axis=1)
        else:
            sensory_words_testdf[acc_axis + temp[0] + temp[1] + temp[2]] = sensory_words_testdf[[
                acc_axis, temp[0], temp[1] + temp[2]]].apply(lambda row: form_words(row), axis=1)

    # for acc_axis in channels[:3]:
    #     temp = []
    #     for gyro_axis in channels[3:]:

    #         if acc_axis[0] != gyro_axis[0]:

    #             temp.append(gyro_axis)
    #     if flag_train:
    #         sensory_words_traindf[acc_axis + temp[0] + temp[1]] = sensory_words_traindf[[acc_axis, temp[0], temp[1]]].apply(lambda row: form_words(row, flag_train), axis=1)
    #     else:
    #         sensory_words_testdf[acc_axis + temp[0] + temp[1]] = sensory_words_testdf[[acc_axis, temp[0], temp[1]]].apply(lambda row: form_words(row), axis=1)


if __name__ == '__main__':

    cluster_cnts = int(sys.argv[1])
    window_length = int(sys.argv[2])
    window_overlap = int(sys.argv[3])
    print(f'Starting generate_subsequences_uci_har  : {cluster_cnts} ')

    train_file_path = os.getcwd() + f'/../../data/output_csv/processed_data_train.csv'
    test_file_path = os.getcwd() + f'/../../data/output_csv/processed_data_test.csv'
    col_names = ['subject_id', 'activityID',
                 'X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2']

    train_df = load_train_test_data(train_file_path, col_names)
    test_df = load_train_test_data(test_file_path, col_names)

    statistics_train = window_sampling(
        train_df, window_length=window_length, window_overlap=window_overlap, flag_train=True)
    statistics_test = window_sampling(
        test_df, window_length=window_length, window_overlap=window_overlap)
    print(
        f'Finished statistics feature extraction  : {cluster_cnts}, {window_length}, {window_overlap} ')

    print(
        f'Starting Clustering  : {cluster_cnts}, {window_length}, {window_overlap} ')
    perform_clustering(statistics_train, statistics_test,
                       channels=col_names[2:], cluster_cnts=cluster_cnts, words_generation_flag=True)
