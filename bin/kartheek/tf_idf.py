import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from sklearn.metrics.pairwise import cosine_similarity


vocab = ['X1_A', 'X1_B', 'X1_C', 'X2_A', 'X2_B', 'Y1_A', 'Y1_B', 'Y1_C', 'Y2_A', 'Z1_A', 'Z1_B', 'Z2_A', 'Z2_C', 'X2_C', 'Y2_B', 'Z2_B', 'Y2_C', 'Z1_C', 'Y1_D']

def load_data(input_txt_filepath):

    # _get all txt files inside file_path
    txt_files = glob.glob(input_txt_filepath)
    #spliting activities document and appending into sample_list
    sample_list = []
    #activites document
    main_docs = []
    activity_list = []
    for txt_file in txt_files:
        start_ind = 0
        tmp_list = []
        activity = txt_file.split('/')[-1].split('_')[-1].split('.')[0]
        activity_list.append('activity_'+str(activity))

        data = (open(txt_file, "r")).read().splitlines()[:1300]

        while start_ind < len(data):
            split_doc_list = []
            end_ind = start_ind + 100
            for doc in data[start_ind:end_ind]:
                split_doc_list.extend(doc.split(' '))
                tmp_list.extend(doc.split(' '))
            sample_list.append(split_doc_list)
            start_ind = end_ind
        main_docs.append(tmp_list)

    return sample_list, activity_list, main_docs


def get_corpus(docs):
    corpus = []
    for sample in docs:
        corpus.append([vocab.index(val) for val in sample])
    return corpus



def tfidf_embeddings(input_txt_filepath):

    samples, activities, activity_docs = load_data(input_txt_filepath)
    documents = get_corpus(activity_docs)
    dct = Dictionary(samples)  # fit dictionary
    corpus = [dct.doc2bow(line) for line in samples]  # convert corpus to BoW format
    model = TfidfModel(corpus)  # fit model
    doc_size = len(samples)
    embeddings = np.zeros((19, doc_size)).tolist()
    for ind in range(len(corpus)):
        # apply model to each document of corpus 
        vector = model[corpus[ind]]
        for tfidf_tuple in vector:
            embeddings[tfidf_tuple[0]][ind] = tfidf_tuple[1]

    cluster_embeddings = np.asarray(embeddings, dtype=float)
    
    return vocab, cluster_embeddings, documents, activities


def plot_vector_similarity(embeddings):

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

    ax.set_title("Cosine similarities of TF-IDF embeddings")
    fig.tight_layout()
    plt.savefig('output/' +
                'embeddings_similarities.png', bbox_inches='tight')
    plt.show()



if __name__ == "__main__":

    input_txt_filepath = os.getcwd() + f'/../../data/sub_sequence_output/*activity*.txt'
    samples, activities_list, main_activity_docs = load_data(input_txt_filepath)
    vocab, cluster_embeddings, documents, activities = tfidf_embeddings(input_txt_filepath)
    plot_vector_similarity(cluster_embeddings)