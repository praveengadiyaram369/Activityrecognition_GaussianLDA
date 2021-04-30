import os
import sys
import csv
import numpy as np
from gaussianlda import GaussianLDAAliasTrainer
from gaussianlda.model import GaussianLDA
from collections import Counter
from statistics import mode
from embeddings_generator import get_cluster_embeddings, get_test_documents
from glda_mapping import get_activity_topic_mapping

from sklearn.metrics import classification_report, f1_score
from sklearn.metrics.cluster import adjusted_rand_score
from tqdm import tqdm


def print_testresults(test_results, classification_report_dict, cluster_cnts, window_length, window_overlap, doc_details):

    accuracy = classification_report_dict['accuracy']*100
    ari = classification_report_dict['adjusted_rand_index_score']*100
    f1_score_macro = classification_report_dict['f1_score_macro']*100
    #weighted_average_precision = classification_report_dict['weighted avg']['precision'] * 100
    #weighted_average_recall = classification_report_dict['weighted avg']['recall'] * 100
    #weighted_average_f1_score = classification_report_dict['weighted avg']['f1-score'] * 100

    glda_output = [window_length, window_overlap, cluster_cnts, accuracy, ari, f1_score_macro]
    doc_details.extend(glda_output)

    with open("output/glda_performance_data.csv", "a", newline='') as fp:
        wr = csv.writer(fp, dialect='excel')
        wr.writerow(doc_details)

    print(f'finished writing results to file : {cluster_cnts} ')

    # for key, val in test_results.items():

    #     print(f'----------- {key} --------------')
    #     total_words = val[1]
    #     for distribution in val[0].most_common():
    #         percentage = (distribution[1]/total_words) * 100
    #         print(f'Topic: {distribution[0]}  = {percentage}')


def get_mode(test_topics):

    counter = Counter(test_topics)
    return counter.most_common()[0][0]

def get_kmeanscluster(test_topics, kmeans_model):

    counter = Counter(test_topics)
    counter_dict = dict(counter)
    topic_dist = np.zeros(6)
    for topic, distribution in counter_dict.items():
        topic_dist[topic] = distribution
    
    div_norm = sum(topic_dist)
    topic_dist = np.divide(topic_dist, div_norm)
    
    return kmeans_model.predict(topic_dist.reshape(1, 6))

if __name__ == "__main__":

    # _for documents
    input_txt_filepath_train = os.getcwd(
    ) + f'/../../data/documents/train/*activity_subseq*.txt'
    input_txt_filepath_test = os.getcwd(
    ) + f'/../../data/documents/test/*activity_subseq*.txt'

    # _for cluster embeddings
    embeddings_filepath = os.getcwd(
    ) + f'/../../data/sub_sequence_output/word_embeddings_from_clusters.json'

    output_dir = "saved_model"
    alpha = 0.01
    cluster_cnts = int(sys.argv[1])
    window_length = int(sys.argv[2])
    window_overlap = int(sys.argv[3])

    vocab, embeddings, corpus, activity_labels, activity_doc_count_index = get_cluster_embeddings(
        input_txt_filepath_train, input_txt_filepath_test, embeddings_filepath, cluster_cnts)

    D = len(corpus)
    N = sum([len(val) for val in corpus])
    V = len(vocab)
    B = int(N/D)

    print(f'Total documents, D: {D}')
    print(f'Total No. of Words, N: {N}')
    print(f'Total Vocabulary, V: {V}')
    print(f'Average No. of words per documents, B: {B}')

    num_topics = len(set(activity_labels))

    # Prepare a trainer
    trainer = GaussianLDAAliasTrainer(
        corpus, embeddings, vocab, num_topics, alpha, save_path=output_dir, kappa=0.3
    )
    print(f'Starting glda clustering training : {cluster_cnts} ')
    # Set training running
    trainer.sample(20, 5)

    activity_topic_mapping = get_activity_topic_mapping(
        list(set(activity_labels)), activity_doc_count_index)

    model = GaussianLDA.load(output_dir)

    test_docs, test_doc_labels = get_test_documents()

    iterations = 25

    test_results = {}

    test_doc_true = []
    test_doc_glda = []

    print(f'Starting glda testing  : {cluster_cnts} ')
    for doc, activity in tqdm(zip(test_docs, test_doc_labels)):
        test_topics = model.sample(doc, iterations)

        true_doc_id = int((activity_topic_mapping[activity])[5:])
        test_doc_true.append(true_doc_id)

        test_doc_glda.append(get_mode(test_topics))
        test_results[activity] = (Counter(test_topics), len(test_topics))

    classification_report_dict = classification_report(
        test_doc_true, test_doc_glda, output_dict=True)
    classification_report_dict['f1_score_macro'] = f1_score(test_doc_true, test_doc_glda, average='macro')
    classification_report_dict['adjusted_rand_index_score'] = adjusted_rand_score(
        test_doc_true, test_doc_glda)

    print_testresults(
        test_results, classification_report_dict, cluster_cnts, window_length, window_overlap, [D, N, V, B])
