import os
import pickle
import json
import pandas as pd
import numpy as np
import joblib

from collections import Counter
from sklearn.preprocessing import Normalizer
from gaussianlda.model import GaussianLDA

global sensory_words
sensory_words = pd.DataFrame()

def window_sampling(main_df, window_length=16, window_overlap=16):

    start_index = 0
    max_window_index, channels = main_df.shape[0], main_df.shape[1]
    column_names = list(main_df.columns)
    sub_sequences = [[] for x in range(channels)]

    while start_index <= (max_window_index - window_length):

        for idx in range(channels):
            sub_sequences[idx].append(main_df[column_names[idx]][start_index:start_index+window_length].tolist())

        start_index += window_overlap

    np_sequences = np.asarray(sub_sequences)

    return  np_sequences, max_window_index, channels, column_names


def raw_clustering(np_sequences):

    raw_clusters = []
    files = os.listdir(os.getcwd()+f'/models/model_subsequences')
    files.sort()
    for idx, file in enumerate(files):

        raw_clusters.append(predict_raw_clustering('models/model_subsequences/'+file, np_sequences[idx]))
    
    return np.array(raw_clusters)


def predict_raw_clustering(file, sub_sequence):

    model = load_model(file)        
    cluster_ids = pd.DataFrame(model.predict(sub_sequence), columns=['cluster ID'])
    one_hot_features_map = {0:[1,0,0,0,0,0], 
                            1:[0,1,0,0,0,0], 
                            2:[0,0,1,0,0,0], 
                            3:[0,0,0,1,0,0], 
                            4:[0,0,0,0,1,0], 
                            5:[0,0,0,0,0,1]}

    one_hot_features_test = [one_hot_features_map[idx] for idx in cluster_ids['cluster ID'].values.tolist()]

    return one_hot_features_test


def load_model(file):

    with open(file, "rb") as f:
        model = pickle.load(f)
    
    return model


def onehotencoding_pooling(subsequences, max_window_index, channels, pooling_size=2):

    pooled_features = [[] for x in range(channels)]
    window_index = 0

    while window_index <= (max_window_index - pooling_size):

        for idx in range(channels):

            pooled_features[idx].append(feature_sum(subsequences[idx][window_index:window_index+pooling_size]))

        window_index += pooling_size
    
    return np.asarray(pooled_features)
    

def feature_sum(vec_list):

    vec_list = np.array(vec_list)
    vec_sum = vec_list[0]
    try:
        for idx in range(1, len(vec_list)):
            vec_sum += vec_list[idx]
        vec_sum = vec_sum/len(vec_list)
    except:
        print('printing exception reason',vec_list)
        exit()
    return vec_sum.tolist()


def pooled_features_clustering(pooled_features, column_names):

    files = os.listdir(os.getcwd()+f'/models/model_pooledfeatures')
    files.sort()
    for idx, file_name in enumerate(zip(files,column_names)):
        
        sensory_words[file_name[1]] = predict_pooledfeatures('models/model_pooledfeatures/'+file_name[0], pooled_features[idx], file_name[1])


def predict_pooledfeatures(file, pooled_features, col_name):

    norm_pooled_features = Normalizer().fit_transform(np.array(pooled_features))
    model = load_joblib_model(file)
    cluster_ids = pd.DataFrame(model.predict(norm_pooled_features), columns=['cluster ID'])
    cluster_ids['cluster ID'] = col_name +'_' + cluster_ids['cluster ID'].astype(str)

    return cluster_ids['cluster ID'].values.tolist()        


def load_joblib_model(file):

    model = joblib.load(file)

    return model


def new_words_generation():

    two_word_combinations, three_word_combinations, four_word_combinations, five_word_combinations, six_word_combinations = False, True, True, True, True
    word_combinations_2 = [['X1', 'Y1'], ['X1', 'Z1'], ['Y1', 'Z1']]
    word_combinations_3 = [['X1', 'Y1', 'Z1'], ['Y1', 'Z1', 'Z2']]
    word_combinations_4 = [['X1', 'Y1', 'Y2', 'Z2'], ['X1', 'Z1', 'X2', 'Y2'], ['Y1', 'Z1', 'X2', 'Z2'], [
        'X1', 'Y1', 'X2', 'Z2'], ['X1', 'Z1', 'Y2', 'Z2'], ['Y1', 'Z1', 'X2', 'Y2']]
    word_combinations_5 = [['X1', 'Y1', 'Z1', 'Y2', 'Z2'], ['X1', 'Y1', 'Z1', 'X2', 'Z2'], [
        'X1', 'Y1', 'Z1', 'X2', 'Z2'], ['X1', 'Y1', 'Z1', 'Y2', 'Z2']]
    word_combinations_6 = [['X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2']]

    if two_word_combinations:
        generate_word_combinations(word_combinations_2)

    if three_word_combinations:
        generate_word_combinations(word_combinations_3)

    if four_word_combinations:
        generate_word_combinations(word_combinations_4)

    if five_word_combinations:
        generate_word_combinations(word_combinations_5)

    if six_word_combinations:
        generate_word_combinations(word_combinations_6)


def generate_word_combinations(word_combinations):

    for combinations in word_combinations:
        
        new_channel_key = ''.join(combinations)
        
        sensory_words[new_channel_key] = sensory_words[combinations].apply(
            lambda row: ''.join(list(row.values.astype(str))), axis=1)


def data_preprocess_pipeline(main_df):

    np_sequences, max_window_index, channels, column_names = window_sampling(main_df)

    raw_features = raw_clustering(np_sequences)

    pooled_features = onehotencoding_pooling(raw_features, raw_features.shape[1], channels)

    pooled_features_clustering(pooled_features, column_names)

    new_words_generation()


def get_mode(test_topics):

    counter = Counter(test_topics)
    return counter.most_common()[0][0]


def glda_clustering(main_df):

    data_preprocess_pipeline(main_df)

    output_dir = os.getcwd()+f'/models/saved_model'

    model = GaussianLDA.load(output_dir)

    input_data = sensory_words.to_numpy().flatten().tolist()

    iterations = 100
    topics = model.sample(input_data, iterations)

    topic_predicted = get_mode(topics)

    # with open('/models/topic-activity.json', 'r') as f:
    #     activity_topic_map = json.load(f)
    
    return topic_predicted