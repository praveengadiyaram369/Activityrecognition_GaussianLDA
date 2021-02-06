# _importing required libraries
import os
import glob
import numpy as np

from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split

train_docs = []
train_doc_labels = []
test_docs = []
test_doc_labels = []


def generate_words(cluster_cnts):

    prefix = 'W_'
    return [prefix+str(val) for val in range(1, cluster_cnts+1)]


def load_data(input_txt_filepath):

    # _get all txt files inside file_path
    txt_files = glob.glob(input_txt_filepath)

    sample_list = []
    activity_list = []

    for txt_file in txt_files:
        tmp_list = []
        activity = txt_file.split('/')[-1].split('_')[-1].split('.')[0]
        activity_list.append('activity_'+str(activity))

        data = (open(txt_file, "r")).read().splitlines()
        for doc in data:
            tmp_list.extend(doc.split(' '))

        sample_list.append(tmp_list)

    for doc, label in zip(sample_list, activity_list):
        train_doc, test_doc = train_test_split(
            doc, test_size=0.25, random_state=1)

        train_docs.append(train_doc)
        test_docs.append(test_doc)

        train_doc_labels.append(label)
        test_doc_labels.append(label)

    return train_docs, train_doc_labels


def get_corpus(vocab, docs):
    corpus = []
    for sample in docs:
        corpus.append([vocab.index(val) for val in sample])
    return corpus


def filter_embeddings(vocab, embeddings, clustering_cnts):

    cluster_names = generate_words(clustering_cnts)

    final_vocab = []
    final_embeddings = []
    for idx, word in enumerate(cluster_names):
        if word in vocab:
            final_vocab.append(word)
            final_embeddings.append(embeddings[idx])

    return final_vocab, final_embeddings


def get_cluster_embeddings(input_txt_filepath, embeddings_filepath, clustering_cnts):

    samples, activities = load_data(input_txt_filepath)

    # total_words = []
    # for sample in samples:
    #     total_words.extend(set(sample))
    # vocab = set(total_words)
    vocab = generate_words(clustering_cnts)

    data = (open(embeddings_filepath, "r")).read().splitlines()
    embeddings = [emb.split(',') for emb in data]

    #vocab_updated, embeddings_updated = filter_embeddings(vocab, embeddings, clustering_cnts)
    cluster_embeddings = np.array(embeddings)
    cluster_embeddings[cluster_embeddings == ''] = '0.0'
    cluster_embeddings = cluster_embeddings.astype(np.float)

    corpus = get_corpus(vocab, samples)

    power = PowerTransformer(method='yeo-johnson', standardize=True)
    cluster_embeddings = power.fit_transform(cluster_embeddings)

    return vocab, cluster_embeddings, corpus, activities


def get_test_documents():
    return test_docs, test_doc_labels
