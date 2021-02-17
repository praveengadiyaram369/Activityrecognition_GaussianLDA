# _importing required libraries
import os
import glob
import numpy as np

from collections import defaultdict
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


def load_data(input_txt_filepath):

    # _get all txt files inside file_path
    txt_files = glob.glob(input_txt_filepath)

    activity_list = []
    activity_doc_count_index = defaultdict(list)
    doc_count = -1

    for txt_file in txt_files:

        tmp_list = []
        activity = txt_file.split('/')[-1].split('_')[-1].split('.')[0]
        label = 'activity_'+str(activity)

        data = (open(txt_file, "r")).read().splitlines()

        for doc in data:
            tmp_list.extend(doc.split(' '))

        # train_doc, test_doc = train_test_split(
        #     tmp_list, test_size=0.25, random_state=1)

        # train_docs.append(train_doc)
        # test_docs.append(test_doc)

        # # train_doc_labels.append(label)
        # test_doc_labels.append(label)

        doc_count += 1
        if label not in activity_doc_count_index:
            activity_doc_count_index[label] = [doc_count]
        else:
            if len(activity_doc_count_index[label]) > 5:
                doc_count -= 1
                test_docs.append(tmp_list)
                test_doc_labels.append(label)
                continue

            activity_doc_count_index[label].append(doc_count)

        train_docs.append(tmp_list)
        activity_list.append('activity_'+str(activity))

    # print(activity_doc_count_index)
    return train_docs, activity_list, activity_doc_count_index


def get_corpus(vocab, docs):
    corpus = []
    for sample in docs:
        corpus.append([vocab.index(val) for val in sample])
    return corpus


def filter_embeddings(vocab, embeddings):

    cluster_cnt = 100
    sequence_names = ['X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2', 'X3', 'Y3', 'Z3']
    cluster_names = generate_cluster_names(sequence_names, cluster_cnt)

    total_clusters = []
    for key, val in cluster_names.items():
        total_clusters.extend(val)

    final_vocab = []
    final_embeddings = []
    for idx, word in enumerate(total_clusters):
        if word in vocab:
            final_vocab.append(word)
            final_embeddings.append(embeddings[idx])

    return final_vocab, final_embeddings


def get_cluster_embeddings(input_txt_filepath, embeddings_filepath):

    samples, activities, activity_doc_count_index = load_data(
        input_txt_filepath)

    # vocab = cluster_vocab.copy()
    # total_words = []
    # for sample in samples:
    #     total_words.extend(set(sample))
    # vocab = set(total_words)

    cluster_cnt = 100
    sequence_names = ['X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2', 'X3', 'Y3', 'Z3']
    vocab = generate_cluster_names(sequence_names, cluster_cnt)

    data = (open(embeddings_filepath, "r")).read().splitlines()
    embeddings = [emb.split(',') for emb in data]

    #vocab_updated, embeddings_updated = filter_embeddings(vocab, embeddings)
    cluster_embeddings = np.array(embeddings)
    cluster_embeddings[cluster_embeddings == ''] = '0.0'
    cluster_embeddings = cluster_embeddings.astype(np.float)

    corpus = get_corpus(vocab, samples)

    return vocab, cluster_embeddings, corpus, activities, activity_doc_count_index


def get_test_documents():
    return test_docs, test_doc_labels
