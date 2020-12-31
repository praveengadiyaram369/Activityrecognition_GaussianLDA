# _importing required libraries
import os
import glob
import collections

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def generate_cluster_names(sequence_names, cluster_cnt=100):

    words_dict = {}

    for seq in sequence_names:
        prefix = seq
        words_dict[seq] = [prefix+'_'+str(i) for i in range(cluster_cnt)]

    return words_dict


def load_data(input_txt_filepath):

    # _get all txt files inside file_path
    txt_files = glob.glob(input_txt_filepath)

    sample_list = []
    activity_list = []

    for txt_file in txt_files:
        tmp_list = []
        activity = txt_file.split('/')[-1].split('_')[-1].split('.')[0]
        activity_list.append('activity_'+str(activity))

        data = (open(txt_file, "r")).read().splitlines()[:3000]
        for doc in data:
            d = doc.split(' ')
            tmp_list.extend(d)

        sample_list.append(tmp_list)

    return sample_list, activity_list


def get_model(samples):
    return Word2Vec(sentences=samples, min_count=1, window=3, size=100)


def get_corpus(vocab, docs):
    corpus = []
    for sample in docs:
        corpus.append([vocab.index(val) for val in sample])
    return corpus


def get_word_embeddings(input_txt_filepath):

    samples, activities = load_data(input_txt_filepath)
    model = get_model(samples)

    vocab = list(model.wv.vocab.keys())
    word_embeddings = model.wv.vectors
    corpus = get_corpus(vocab, samples)

    return vocab, word_embeddings, corpus, activities


def filter_embeddings(vocab, embeddings):

    cluster_cnt = 00
    sequence_names = ['X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2']
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

    samples, activities = load_data(input_txt_filepath)

    # vocab = cluster_vocab.copy()
    total_words = []
    for sample in samples:
        total_words.extend(set(sample))
    vocab = set(total_words)
    data = (open(embeddings_filepath, "r")).read().splitlines()
    embeddings = [emb.split(',') for emb in data]

    vocab_updated, embeddings_updated = filter_embeddings(vocab, embeddings)
    cluster_embeddings = np.array(embeddings_updated)
    cluster_embeddings[cluster_embeddings == ''] = '0.0'
    cluster_embeddings = cluster_embeddings.astype(np.float)

    corpus = get_corpus(vocab_updated, samples)

    return vocab_updated, cluster_embeddings, corpus, activities


def plot_vector_similarity(vocab, embeddings):

    vocab_len = len(vocab)
    cos_sims = np.zeros(shape=(vocab_len, vocab_len))
    for vec_1 in range(vocab_len):
        x = np.array(embeddings[vec_1]).reshape(1, -1)
        for vec_2 in range(vocab_len):
            cos_sims[vec_1][vec_2] = cosine_similarity(
                x, np.array(embeddings[vec_2]).reshape(1, -1))

    fig, ax = plt.subplots()
    im = ax.imshow(cos_sims)

    ax.set_xticks(np.arange(vocab_len))
    ax.set_yticks(np.arange(vocab_len))

    ax.set_xticklabels(vocab)
    ax.set_yticklabels(vocab)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    for i in range(vocab_len):
        for j in range(vocab_len):
            text = ax.text(j, i, str(cos_sims[i, j] * 100)[:4],
                           ha="center", va="center", color="w")

    ax.set_title("Cosine similarities of word2vec embeddings")
    fig.tight_layout()
    plt.savefig('output/' +
                'embeddings_similarities.png', bbox_inches='tight')
    plt.show()


def plot_pca(vocab):

    # fit a 2d PCA model to the vectors
    X = model[vocab]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    # create a scatter plot of the projection
    plt.scatter(result[:, 0], result[:, 1])
    words = list(vocab)
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))

    plt.savefig('output/' +
                'embeddings_pca_results.png', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":

    # _manual testing purpose
    input_txt_filepath = os.getcwd() + f'/../../data/sub_sequence_output/*activity*.txt'
    samples, activities = load_data(input_txt_filepath)
    # print('#####',activities)

    # model = get_model(samples)
    # print(model)

    #vocab = list(model.wv.vocab.keys())
    # print(vocab)
    # print(model.wv.vectors)

    # plot_pca(model.wv.vocab)
    plt.rcParams["figure.figsize"] = [16, 9]
    #plot_vector_similarity(vocab, model.wv.vectors)
    # data = (open(os.getcwd(
    # ) + f'/../../data/sub_sequence_output/word_embeddings_from_clusters.txt', "r")).read().splitlines()
    # embeddings = [emb.split(',') for emb in data]
    # plot_vector_similarity(cluster_vocab, embeddings)
