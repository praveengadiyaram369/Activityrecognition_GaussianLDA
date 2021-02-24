# _importing required libraries
import os
import glob
import numpy as np

from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from gensim import corpora, models
import matplotlib.pyplot as plt

train_docs = []
train_doc_labels = []
test_docs = []
test_doc_labels = []


def plot_idfdata(tfidf_values):

    x = range(len(tfidf_values))
    y = sorted(tfidf_values, reverse=True)

    plt.scatter(x, y, marker='.')
    plt.xlabel('words')
    plt.ylabel('tf-idf values')
    plt.savefig('output/tfidf_plot.png', bbox_inches='tight')


def filter_embeddings(vocab, embeddings, stop_words):

    final_vocab = []
    final_embeddings = []
    for idx, word in enumerate(vocab):
        if word not in stop_words:
            final_vocab.append(word)
            final_embeddings.append(embeddings[idx])

    return final_vocab, np.array(final_embeddings)

def filter_documents(stop_words, train_or_test_flag):

    if train_or_test_flag == True:
        docs = train_docs
    else:
        docs = test_docs

    docs_updated = [list(filter(lambda x:x not in stop_words, doc)) for doc in docs]    
    return docs_updated

def get_stop_words():

    dictionary = corpora.Dictionary(train_docs)

    corpus = [dictionary.doc2bow(doc) for doc in train_docs]
    tfidf = models.TfidfModel(corpus, smartirs='ntc')

    tfidf_dict = {}
    for doc in tfidf[corpus]:
        for id, freq in doc:
            word = dictionary[id]
            value = np.around(freq, decimals=2)
            if word not in tfidf_dict.keys():
                tfidf_dict[word] = value

    plot_idfdata(tfidf_dict.values())

    idf_threshold = 0.2
    stop_words = []
    tfidf_dict_sorted = sorted(tfidf_dict.items(), key=lambda x: x[1])

    for word in tfidf_dict_sorted:
        if word[1] < idf_threshold:
            stop_words.append(word[0])
        else:
            break

    return stop_words


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
            # if len(activity_doc_count_index[label]) > 400:
            #     continue

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

    return train_doc_labels, activity_doc_count_index


def get_corpus(vocab):
    vocab = list(vocab)
    corpus = []
    for sample in train_docs:
        corpus.append([vocab.index(val) for val in sample])
    return corpus


def get_embeddings(embeddings_filepath):

    data = (open(embeddings_filepath, "r")).read().splitlines()
    embeddings = [emb.split(',') for emb in data]

    cluster_embeddings = np.array(embeddings)
    cluster_embeddings[cluster_embeddings == ''] = '0.0'
    cluster_embeddings = cluster_embeddings.astype(np.float)
    cluster_embeddings = Normalizer().fit_transform(cluster_embeddings)

    return cluster_embeddings


def get_cluster_embeddings(input_txt_filepath_train, input_txt_filepath_test, embeddings_filepath):

    activities, activity_doc_count_index = load_data(
        input_txt_filepath_train, train_or_test_flag=True)
    load_data(input_txt_filepath_test, train_or_test_flag=False)

    cluster_cnt = 250
    sequence_names = ['X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2']
    vocab = generate_cluster_names(sequence_names, cluster_cnt)
    cluster_embeddings = get_embeddings(embeddings_filepath)

    assert len(vocab) == len(cluster_embeddings)

    stop_words = get_stop_words()
    np.savetxt('output/stopwords.txt', np.array(stop_words), delimiter=',', fmt='%5s')
    print(f'No. of Stop words: {len(stop_words)} \n')

    vocab, cluster_embeddings = filter_embeddings(
        vocab, cluster_embeddings, stop_words)

    assert len(vocab) == cluster_embeddings.shape[0]

    global train_docs
    global test_docs

    train_docs = filter_documents(stop_words, 
        train_or_test_flag=True)
    test_docs = filter_documents(stop_words, train_or_test_flag=False)
    corpus = get_corpus(vocab)

    return vocab, cluster_embeddings, corpus, activities, activity_doc_count_index


def get_test_documents():
    return test_docs, test_doc_labels
