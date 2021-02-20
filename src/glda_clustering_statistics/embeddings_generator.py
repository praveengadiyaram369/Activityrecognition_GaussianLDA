# _importing required libraries
import os
import glob
import numpy as np

from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split

train_docs = []
train_doc_labels = []
test_docs = []
test_doc_labels = []


def generate_cluster_names(sequence_names, cluster_cnt=100):

    vocab = []

    for seq in sequence_names:
        prefix = seq
        vocab.extend([prefix+'_'+str(i) for i in range(cluster_cnt)])

    return vocab


def load_data(input_txt_filepath, train_or_test_flag=True):

    # _get all txt files inside file_path
    txt_files = glob.glob(input_txt_filepath)

    train_doc_labels = []
    activity_doc_count_index = defaultdict(list)
    doc_count = -1

    for txt_file in txt_files:

        tmp_list = []
        activity = txt_file.split('/')[-1].split('_')[-1].split('.')[0]
        label = 'activity_'+str(activity)

        data = (open(txt_file, "r")).read().splitlines()
        for doc in data:
            tmp_list.extend(doc.split(' '))

        if train_or_test_flag:
            
            doc_count += 1
            if label not in activity_doc_count_index:
                activity_doc_count_index[label] = [doc_count]
            else:
                activity_doc_count_index[label].append(doc_count)

            train_docs.append(tmp_list)
            train_doc_labels.append(label)
        else:
            test_docs.append(tmp_list)
            test_doc_labels.append(label)

    if train_or_test_flag == False:
        return None

    return train_docs, train_doc_labels, activity_doc_count_index


def get_corpus(vocab, docs):
    corpus = []
    for sample in docs:
        corpus.append([vocab.index(val) for val in sample])
    return corpus


def get_cluster_embeddings(input_txt_filepath_train, input_txt_filepath_test, embeddings_filepath):

    samples, activities, activity_doc_count_index = load_data(
        input_txt_filepath_train, train_or_test_flag=True)
    load_data(input_txt_filepath_test, train_or_test_flag=False)

    cluster_cnt = 250
    sequence_names = ['X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2']
    vocab = generate_cluster_names(sequence_names, cluster_cnt)

    data = (open(embeddings_filepath, "r")).read().splitlines()
    embeddings = [emb.split(',') for emb in data]

    cluster_embeddings = np.array(embeddings)
    cluster_embeddings[cluster_embeddings == ''] = '0.0'
    cluster_embeddings = cluster_embeddings.astype(np.float)

    corpus = get_corpus(vocab, samples)

    return vocab, cluster_embeddings, corpus, activities, activity_doc_count_index


def get_test_documents():
    return test_docs, test_doc_labels
