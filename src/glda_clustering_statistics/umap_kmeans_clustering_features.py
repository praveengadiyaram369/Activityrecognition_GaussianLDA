import os
import sys
import umap
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.cluster import KMeans

def get_assigned_words(seq_clusters, cluster_words):
    
    # _assign word to each cluster of the subsequence usnig numpy where function
    assigned_words = np.where(seq_clusters != 0, seq_clusters, cluster_words[0])
    for idx in range(1, len(cluster_words)):
         assigned_words = np.where(seq_clusters != idx, assigned_words, cluster_words[idx])
                
    return assigned_words

def get_cluster_names(prefix):

    words = [prefix+'_'+str(i) for i in range(cluster_cnts)]
    return words

def get_umap_data(data):
    
    data = Normalizer().fit_transform(data)
    model_umap = umap.UMAP(n_neighbors = 50, min_dist = 0.1, n_components = 5)
    umap_vec = model_umap.fit_transform(data)

    featureVec = Normalizer().fit_transform(umap_vec)
    return featureVec

def process_channel_data(channel_data_train, channel_data_test, prefix):
    
    channel_data_train = get_umap_data(channel_data_train)
    cluster_names = get_cluster_names(prefix)
    
    model = KMeans(n_clusters=cluster_cnts, random_state=2).fit(channel_data_train)
    seq_clusters = model.predict(channel_data_train)
    assigned_clusterWord = get_assigned_words(seq_clusters, cluster_names)
    doc_df[prefix] = assigned_clusterWord
    
    clusters.extend(cluster_names)
    cluster_centers.extend(model.cluster_centers_)
    
    channel_data_test = get_umap_data(channel_data_test)
    seq_clusters = model.predict(channel_data_test)
    assigned_clusterWord = get_assigned_words(seq_clusters, cluster_names)
    doc_test_df[prefix] = assigned_clusterWord

if __name__ == '__main__':

    cluster_cnts = int(sys.argv[1])
    current_path = os.getcwd()
    subject_activity_data = np.loadtxt(current_path+'/../../data/lstm_features/activity_subject_data.csv', delimiter=',')
    sensor_features = np.loadtxt(current_path+'/../../data/lstm_features/UCIHAR_sensor_features_lstm_tuned_train_16.csv', delimiter=',')

    doc_df = pd.DataFrame(columns=['subjectID', 'activityID'])
    doc_df['subjectID'] = subject_activity_data[:,0].astype(int)
    doc_df['activityID'] = subject_activity_data[:,1].astype(int)
    doc_df

    subject_activity_data_test = np.loadtxt(current_path+'/../../data/lstm_features/activity_subject_data_test.csv', delimiter=',')
    sensor_features_test = np.loadtxt(current_path+'/../../data/lstm_features/UCIHAR_sensor_features_lstm_tuned_test_16.csv', delimiter=',')

    doc_test_df = pd.DataFrame(columns=['subjectID', 'activityID'])
    doc_test_df['subjectID'] = subject_activity_data_test[:,0].astype(int)
    doc_test_df['activityID'] = subject_activity_data_test[:,1].astype(int)
    doc_test_df

    channel_names = ['X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2']
    step_count_train = (7352*15)
    step_count_test = (2947*15)
    
    clusters = []
    cluster_centers = []

    for val in range(6):
        channel_id = val
        prefix = channel_names[channel_id]
        
        channel_data_train = sensor_features[(val*step_count_train):((val+1)*step_count_train), :]
        channel_data_test = sensor_features_test[(val*step_count_test):((val+1)*step_count_test), :]
        
        process_channel_data(channel_data_train, channel_data_test, prefix)
        print(f'clustering channel finished {prefix}')

    assert len(clusters) == len(cluster_centers)
    embeddings_filepath = os.getcwd() + f'/../../data/sub_sequence_output/word_embeddings_from_clusters.txt'
    np.savetxt(embeddings_filepath, np.array(cluster_centers), delimiter= ',')

    #doc_df['activityID'] = doc_df['activityID'].astype(int)
    doc_df['final_sub_sequence'] = doc_df[channel_names].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    for subject in doc_df['subjectID'].unique():
        
        activity = doc_df.loc[(doc_df['subjectID'] == subject)]['activityID'].values[0]
        output_filepath = os.getcwd() + f'/../../data/documents/train/activity_subseq_' + str(subject) +'_'+ str(activity) + '.txt'
        doc_df.loc[(doc_df['subjectID'] == subject)][['final_sub_sequence']].to_csv(output_filepath, sep='\t', index=False, header= False)


    doc_test_df['final_sub_sequence'] = doc_test_df[channel_names].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    for subject in doc_test_df['subjectID'].unique():
        
        activity = doc_test_df.loc[(doc_test_df['subjectID'] == subject)]['activityID'].values[0]
        output_filepath = os.getcwd() + f'/../../data/documents/test/activity_subseq_' + str(subject) +'_'+ str(activity) + '.txt'
        doc_test_df.loc[(doc_test_df['subjectID'] == subject)][['final_sub_sequence']].to_csv(output_filepath, sep='\t', index=False, header= False)