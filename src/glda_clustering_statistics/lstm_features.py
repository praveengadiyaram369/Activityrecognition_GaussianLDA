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
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
import statistics

from global_settings import *
from features_classification import perform_classification_on_features, perform_classification_on_rawfeatures

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

    model = KMeans(n_clusters=cluster_cnts,
                   random_state=234).fit(statistic_train_df)

    cluster_ids = pd.DataFrame(model.predict(
        statistic_train_df), columns=['cluster ID'])
    cluster_test_ids = pd.DataFrame(model.predict(
        statistic_test_df), columns=['cluster ID'])

    seq_clusters = cluster_ids.to_numpy()
    assigned_clusterWord = get_assigned_words(
        seq_clusters, cluster_words, axis, flag_train=True)
    get_assigned_words(cluster_test_ids.to_numpy(), cluster_words, axis)

    centroids_of_clusters = pd.DataFrame(model.cluster_centers_[cluster_ids['cluster ID']],
                                         columns=[f'dim_{val}' for val in range(statistic_train_df.shape[1])])
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

    centroid_statistic = []

    words_dict = generate_cluster_names(channels, cluster_cnts)

    for statistic_train_df, statistic_test_df, axis in zip(statistics_train, statistics_test, channels):

        cluster_names = words_dict[axis]
        axis_clusters = clustering(
            statistic_train_df, statistic_test_df, axis, cluster_cnts, cluster_names)

        for j in range(len(cluster_names)):

            cluster_stats = cluster_word_sort(axis_clusters, cluster_names[j])
            centroid_statistic.append(cluster_stats)
            words_embedding_dict[cluster_names[j]
                                 ] = cluster_stats.values[0].tolist()

    replace_leastidf_flag = False
    if replace_leastidf_flag:
        replace_leastidf_values()

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

    print(f'Finished clustering  : {cluster_cnts} ')

    perform_classification_on_features()


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

    temp = list(row.values.astype(str))
    if flag_train:
        dict_key = ''.join(temp)

        for _ in range(len(words_embedding_dict[temp[0]])):
            vec_list = []
            for val in temp:
                vec_list.append(words_embedding_dict[val])

        words_embedding_dict[dict_key] = feature_sum(vec_list)

    return ''.join(temp)


def generate_word_combinations(word_combinations, flag_train):

    for combinations in word_combinations:
        #acc_axis = combinations[0]
        #temp = combinations[1:]
        new_channel_key = ''.join(combinations)
        if flag_train:
            sensory_words_traindf[new_channel_key] = sensory_words_traindf[combinations].apply(
                lambda row: form_words(row, flag_train), axis=1)
        else:
            sensory_words_testdf[new_channel_key] = sensory_words_testdf[combinations].apply(
                lambda row: form_words(row), axis=1)


def new_words_generation(channels, flag_train=False):

    two_word_combinations, three_word_combinations, four_word_combinations, five_word_combinations, six_word_combinations = False, True, True, True, False
    word_combinations_2 = [['X1', 'Y1'], ['X1', 'Z1'], ['Y1', 'Z1'], ['X1', 'Y2'], ['X1', 'Z2'], ['Y1', 'X2'], ['Y1', 'Z2'], ['Z1', 'X2'], ['Z1', 'Y2']]
    word_combinations_3 = [['X1', 'Y1', 'Z1'], ['X1', 'Y2', 'Z2'], ['Y1', 'X2', 'Z2'], ['Z1', 'X2', 'Y2']]
    word_combinations_4 = [['X1', 'Y1', 'Y2', 'Z2'], ['X1', 'Z1', 'X2', 'Y2'], ['Y1', 'Z1', 'X2', 'Z2'], [
        'X1', 'Y1', 'X2', 'Z2'], ['X1', 'Z1', 'Y2', 'Z2'], ['Y1', 'Z1', 'X2', 'Y2']]
    word_combinations_5 = [['X1', 'Y1', 'Z1', 'Y2', 'Z2'], ['X1', 'Y1', 'Z1', 'X2', 'Z2'], [
        'X1', 'Y1', 'Z1', 'X2', 'Z2'], ['X1', 'Y1', 'Z1', 'Y2', 'Z2']]
    word_combinations_6 = [['X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2']]

    if two_word_combinations:
        generate_word_combinations(word_combinations_2, flag_train)

    if three_word_combinations:
        generate_word_combinations(word_combinations_3, flag_train)

    if four_word_combinations:
        generate_word_combinations(word_combinations_4, flag_train)

    if five_word_combinations:
        generate_word_combinations(word_combinations_5, flag_train)

    if six_word_combinations:
        generate_word_combinations(word_combinations_6, flag_train)


def get_replacement_word(channel_values, replaceword):

    if channel_values[1] == replaceword:
        return channel_values[0]
    return channel_values[1]


def replace_leastidf_values():

    Y2_least_idf = collections.Counter(
        sensory_words_traindf['Y2'].tolist()).most_common()[0][0]
    Z2_least_idf = collections.Counter(
        sensory_words_traindf['Z2'].tolist()).most_common()[0][0]

    sensory_words_traindf['Y2'] = sensory_words_traindf[['Y1', 'Y2']].apply(
        lambda row: get_replacement_word(row.values, Y2_least_idf), axis=1)

    sensory_words_traindf['Z2'] = sensory_words_traindf[['Z1', 'Z2']].apply(
         lambda row: get_replacement_word(row.values, Z2_least_idf), axis=1)

    sensory_words_testdf['Y2'] = sensory_words_testdf[['Y1', 'Y2']].apply(
        lambda row: get_replacement_word(row.values, Y2_least_idf), axis=1)

    sensory_words_testdf['Z2'] = sensory_words_testdf[['Z1', 'Z2']].apply(
         lambda row: get_replacement_word(row.values, Z2_least_idf), axis=1)


def feature_sum(vec_list):
    vec_list = np.array(vec_list)
    vec_sum = vec_list[0]
    try:
        for idx in range(1, len(vec_list)):
            vec_sum += vec_list[idx]
    except:
        print(vec_list)
        exit()
    return vec_sum.tolist()


def mean_feature_sum(vec_list):
    n = len(vec_list)
    vec_list_sum = np.array(feature_sum(vec_list))
    vec_list_sum = vec_list_sum/n
    return vec_list_sum.tolist()


def get_kmeans_clusters(sub_sequence_train, sub_sequence_test, feature_dim):

    model = KMeans(n_clusters=feature_dim,
                   random_state=5).fit(sub_sequence_train)
    cluster_train_ids = pd.DataFrame(model.predict(
        sub_sequence_train), columns=['cluster ID'])
    cluster_test_ids = pd.DataFrame(model.predict(
        sub_sequence_test), columns=['cluster ID'])

    one_hot_features_train = pd.get_dummies(cluster_train_ids.astype('str'))
    one_hot_features_test = pd.get_dummies(cluster_test_ids.astype('str'))

    return one_hot_features_train.values.tolist(), one_hot_features_test.values.tolist()


def extract_feature_info_lstmdata(label_cnt, step_cnt, features, subject_activity_data):

    X = []
    y = []

    for idx in range(label_cnt):
        class_label = subject_activity_data[idx*step_cnt][1]
        y.append(class_label)

        lower_lim = idx*step_cnt
        upper_lim = ((idx+1)*step_cnt)
        temp = []

        for val in range(6):
            temp.append(mean_feature_sum(features[val,lower_lim:upper_lim,:]))

        X.append(mean_feature_sum(temp))

    return X, y


def perform_clf(features_train, features_test, subject_activity_data_train, subject_activity_data_test):

    train_label_cnt = 7352
    test_label_cnt = 2947
    feature_dim = features_train.shape[2]
    step_cnt = int(features_train.shape[1]/train_label_cnt)

    X_train, y_train = extract_feature_info_lstmdata(train_label_cnt, step_cnt, features_train, subject_activity_data_train)
    X_test, y_test = extract_feature_info_lstmdata(test_label_cnt, step_cnt, features_test, subject_activity_data_test)

    X_train = np.array(X_train).reshape(-1,feature_dim).astype('float32')
    y_train = np.array(y_train).astype('int32')
    X_test = np.array(X_test).reshape(-1,feature_dim).astype('float32')
    y_test = np.array(y_test).astype('int32')

    perform_classification_on_rawfeatures(X_train, y_train, X_test, y_test)


if __name__ == '__main__':

    print('\n\n\n')
    print('Starting lstm features....')

    cluster_cnts = int(sys.argv[1])
    window_length = 16
    window_overlap = 8

    subject_activity_data_train = np.loadtxt(os.getcwd() + '/../../data/lstm_data/activity_subject_data_train.csv', delimiter=',')
    sensor_features_train = np.loadtxt(os.getcwd() + '/../../data/lstm_data/UCIHAR_sensor_features_lstm_tuned_train.csv', delimiter=',')
    subject_activity_data_test = np.loadtxt(os.getcwd() + '/../../data/lstm_data/activity_subject_data_test.csv', delimiter=',')
    sensor_features_test = np.loadtxt(os.getcwd() + '/../../data/lstm_data/UCIHAR_sensor_features_lstm_tuned_test.csv', delimiter=',')

    col_names = ['subject_id', 'activityID',
                 'X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2']

    sensory_words_traindf['subject_id'] = subject_activity_data_train[:,0].astype(int)
    sensory_words_traindf['activityID'] = subject_activity_data_train[:,1].astype(int)

    sensory_words_testdf['subject_id'] = subject_activity_data_test[:,0].astype(int)
    sensory_words_testdf['activityID'] = subject_activity_data_test[:,1].astype(int)

    train_channel_len = int(sensor_features_train.shape[0]/6)
    test_channel_len = int(sensor_features_test.shape[0]/6)
    feature_dim = sensor_features_train.shape[1]

    features_train = sensor_features_train.reshape(6, train_channel_len, feature_dim)
    features_test = sensor_features_test.reshape(6, test_channel_len, feature_dim)

    perform_clf(features_train, features_test, subject_activity_data_train, subject_activity_data_test)

    perform_clustering(features_train, features_test,
                       channels=col_names[2:], cluster_cnts=cluster_cnts, words_generation_flag=True)

    print('Ending lstm features....')
