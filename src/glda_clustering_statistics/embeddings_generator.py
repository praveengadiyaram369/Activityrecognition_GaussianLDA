# _importing required libraries
import os
import glob
import numpy as np
import pickle

from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from gensim import corpora, models
import matplotlib.pyplot as plt

train_docs = []
train_doc_labels = []
test_docs = []
test_doc_labels = []
cluster_cnt = 250
sequence_names = ['X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2']
clusters_channelwise = []


def plot_idfdata(tfidf_values):

    y = range(len(tfidf_values))
    x = sorted(tfidf_values, reverse=True)

    plt.scatter(x, y, marker='.')
    plt.ylabel('words')
    plt.xlabel('tf-idf values')
    plt.savefig('output/tfidf_plot.png', bbox_inches='tight')


def filter_embeddings(vocab, embeddings, stop_words):

    final_vocab = []
    final_embeddings = []
    for idx, word in enumerate(vocab):
        if word not in stop_words:
            final_vocab.append(word)
            final_embeddings.append(embeddings[idx])

    return final_vocab, np.array(final_embeddings)


def filter_documents(stop_words):

    global train_docs
    global test_docs

    for idx in range(len(train_docs)):
        train_docs[idx] = list(
            filter(lambda x: x not in stop_words, train_docs[idx]))

    for idx in range(len(test_docs)):
        test_docs[idx] = list(
            filter(lambda x: x not in stop_words, test_docs[idx]))


def update_documents(new_vocab):

    global train_docs
    global test_docs

    new_vocab = [(word.split('+')[0], word.split('+')[1])
                 for word in new_vocab]

    for idx in range(len(train_docs)):
        new_words = []
        for word in new_vocab:
            if word[0] in train_docs[idx] and word[1] in train_docs[idx]:
                new_words.append(f'{word[0]}+{word[1]}')

        if len(new_words) > 0:
            train_docs[idx].extend(new_words)

    for idx in range(len(test_docs)):
        new_words = []
        for word in new_vocab:
            if word[0] in test_docs[idx] and word[1] in test_docs[idx]:
                new_words.append(f'{word[0]}+{word[1]}')

        if len(new_words) > 0:
            test_docs[idx].extend(new_words)


def get_stop_words_fromfile():

    with open(os.getcwd() + f'/../../data/stopwords.pkl', "rb") as doc:
        stop_words = pickle.load(doc)

    print(f'No. of stopwords removed: {len(stop_words)}')

    return stop_words


def get_stop_words(idf_threshold):

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

    stop_words = []
    tfidf_dict_sorted = sorted(tfidf_dict.items(), key=lambda x: x[1])

    for word in tfidf_dict_sorted:
        if word[1] < idf_threshold:
            stop_words.append(word[0])
        else:
            break

    print(f'No. of stopwords removed: {len(stop_words)}')

    return stop_words, tfidf_dict_sorted


def get_euclidean_distance_similarity(vec1, vec2):
    return 1 - np.linalg.norm(vec1-vec2)


def get_new_featurevector(vec1, vec2):
    return (vec1 + vec2)/2


def get_new_vocabulary_from_channels(vocab, cluster_embeddings):

    word_dictionary = {}
    for key, val in zip(vocab, cluster_embeddings):
        word_dictionary[key] = val

    combination_pairs = [('X1', 'Y2'), ('X1', 'Z2'),
                         ('Y1', 'X2'), ('Y1', 'Z2'), ('Z1', 'X2'), ('Z1', 'Y2')]

    new_vocabulary_dict = {}
    for pair in combination_pairs:

        channel_1_clusters = clusters_channelwise[sequence_names.index(
            pair[0])]
        channel_2_clusters = clusters_channelwise[sequence_names.index(
            pair[0])]

        for val1 in range(len(channel_1_clusters)):

            if channel_1_clusters[val1] in word_dictionary.keys():
                vec1 = channel_1_clusters[val1]

                for val2 in range(val1):

                    if channel_2_clusters[val2] in word_dictionary.keys():
                        vec2 = channel_2_clusters[val2]

                        similarity = get_euclidean_distance_similarity(
                            word_dictionary[vec1], word_dictionary[vec2])
                        if similarity > 0.7:
                            new_vocab = f'{vec1}+{vec2}'
                            new_vocabulary_dict[new_vocab] = get_new_featurevector(
                                word_dictionary[vec1], word_dictionary[vec2])

        print(f'No. of new vocab created: {len(new_vocabulary_dict)}')

        return new_vocabulary_dict


def get_new_vocabulary(vocab, cluster_embeddings, tfidf_dict_sorted, idf_threshold=0.3):

    word_dictionary = {}
    for key, val in zip(vocab, cluster_embeddings):
        word_dictionary[key] = val

    top_words = []
    for word in tfidf_dict_sorted:
        if word[1] > idf_threshold:
            top_words.append(word[0])

    new_vocabulary_dict = {}
    for val1 in range(len(top_words)):
        vec1 = top_words[val1]
        for val2 in range(val1):
            if val1 != val2:
                vec2 = top_words[val2]

                similarity = get_euclidean_distance_similarity(
                    word_dictionary[vec1], word_dictionary[vec2])
                if similarity > 0:
                    new_vocab = f'{vec1}+{vec2}'
                    new_vocabulary_dict[new_vocab] = get_new_featurevector(
                        word_dictionary[vec1], word_dictionary[vec2])

    print(f'No. of new vocab created: {len(new_vocabulary_dict)}')

    return new_vocabulary_dict


def generate_cluster_names():

    vocab = []
    global clusters_channelwise

    for seq in sequence_names:
        cluster_names = [seq+'_'+str(i) for i in range(cluster_cnt)]
        clusters_channelwise.append(cluster_names)
        vocab.extend(cluster_names)

    return vocab


def load_data(input_txt_filepath, train_or_test_flag=True):

    # _get all txt files inside file_path
    txt_files = glob.glob(input_txt_filepath)

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

    return activity_doc_count_index


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
    # cluster_embeddings = cluster_embeddings[:, [0, 1]]
    cluster_embeddings[cluster_embeddings == ''] = '0.0'
    cluster_embeddings = cluster_embeddings.astype(np.float)
    cluster_embeddings = Normalizer().fit_transform(cluster_embeddings)

    return cluster_embeddings


def get_cluster_embeddings(input_txt_filepath_train, input_txt_filepath_test, embeddings_filepath):

    reset_global_data()
    stop_words_new_vocab_flag = True
    activity_doc_count_index = load_data(
        input_txt_filepath_train, train_or_test_flag=True)
    load_data(input_txt_filepath_test, train_or_test_flag=False)

    vocab = generate_cluster_names()
    cluster_embeddings = get_embeddings(embeddings_filepath)

    assert len(vocab) == len(cluster_embeddings)

    if stop_words_new_vocab_flag == True:

        # stop_words, tfidf_dict_sorted = get_stop_words(idf_threshold=0.15)
        stop_words = get_stop_words_fromfile()

        vocab, cluster_embeddings = filter_embeddings(
            vocab, cluster_embeddings, stop_words)

        assert len(vocab) == cluster_embeddings.shape[0]

        filter_documents(stop_words)

        # new_vocabulary_dict = get_new_vocabulary(
        #     vocab, cluster_embeddings, tfidf_dict_sorted, idf_threshold=0.26)

        new_vocabulary_dict = get_new_vocabulary_from_channels(
            vocab, cluster_embeddings)

        for new_vocab, feature in new_vocabulary_dict.items():
            vocab.append(new_vocab)
            cluster_embeddings = np.vstack([cluster_embeddings, feature])

        assert len(vocab) == cluster_embeddings.shape[0]

        update_documents(list(new_vocabulary_dict.keys()))

    corpus = get_corpus(vocab)

    return vocab, cluster_embeddings, corpus, train_doc_labels, activity_doc_count_index


def get_test_documents():
    return test_docs, test_doc_labels


def reset_global_data():

    global train_docs
    global train_doc_labels
    global test_docs
    global test_doc_labels

    train_docs = []
    train_doc_labels = []
    test_docs = []
    test_doc_labels = []
