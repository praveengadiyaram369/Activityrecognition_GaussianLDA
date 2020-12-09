# _importing required libraries
import os
import glob

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot


def load_data(input_txt_filepath):

    # _get all txt files inside file_path
    txt_files = glob.glob(input_txt_filepath)

    sample_list = []
    activity_list = []
    for txt_file in txt_files:
        activity = txt_file.split('/')[-1].split('_')[-1].split('.')[0]

        data = (open(txt_file, "r")).read().splitlines()
        for doc in data:
            sample_list.append(doc.split(' '))
            activity_list.append('activity_'+str(activity))

    return sample_list, activity_list


def get_model(samples):
    return Word2Vec(sentences=samples, min_count=1, window=5, size=100)


def get_word_embeddings(input_txt_filepath):

    samples, activities = load_data(input_txt_filepath)
    model = get_model(samples)

    vocab = list(model.wv.vocab.keys())
    word_embeddings = model.wv.vectors
    corpus = []
    for sample in samples:
        corpus.append([vocab.index(val) for val in sample])

    return vocab, word_embeddings, corpus, activities


if __name__ == "__main__":

    # _manual testing purpose
    input_txt_filepath = os.getcwd() + f'/../../data/sub_sequence_output/*.txt'
    samples, activities = load_data(input_txt_filepath)

    model = get_model(samples)
    print(model)

    vocab_glda = model.wv.vocab
    print(list(vocab_glda.keys()))

    print(model.wv.vectors)

    # fit a 2d PCA model to the vectors
    X = model[model.wv.vocab]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    # create a scatter plot of the projection
    pyplot.scatter(result[:, 0], result[:, 1])
    words = list(model.wv.vocab)
    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))

    pyplot.savefig('output/' +
                   'embeddings_pca_results.png', bbox_inches='tight')
    pyplot.show()
