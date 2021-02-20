import os
import csv
import numpy as np
from gaussianlda import GaussianLDAAliasTrainer
from gaussianlda.model import GaussianLDA
from collections import Counter
from statistics import mode
from embeddings_generator import get_cluster_embeddings, get_test_documents
from glda_mapping import get_activity_topic_mapping

from sklearn.metrics import classification_report
from sklearn.metrics.cluster import adjusted_rand_score


def print_testresults(test_results, classification_report_dict):

    accuracy = classification_report_dict['accuracy']*100
    ari = classification_report_dict['adjusted_rand_index_score']
    weighted_average_precision = classification_report_dict['weighted avg']['precision'] * 100
    weighted_average_recall = classification_report_dict['weighted avg']['recall'] * 100
    weighted_average_f1_score = classification_report_dict['weighted avg']['f1-score'] * 100

    glda_output = [accuracy, ari,
                   weighted_average_precision, weighted_average_recall, weighted_average_f1_score]

    with open("output/glda_performance_data.csv", "a", newline='') as fp:
        wr = csv.writer(fp, dialect='excel')
        wr.writerow(glda_output)

    logger.info(
        f'Finished writing glda performance metrics to the output')

    # for key, val in test_results.items():

    #     print(f'----------- {key} --------------')
    #     total_words = val[1]
    #     for distribution in val[0].most_common():
    #         percentage = (distribution[1]/total_words) * 100
    #         print(f'Topic: {distribution[0]}  = {percentage}')


def get_mode(test_topics):

    counter = Counter(test_topics)
    return counter.most_common()[0][0]


if __name__ == "__main__":

    # _for documents
    input_txt_filepath_train = os.getcwd(
    ) + f'/../../data/documents/train/*activity_subseq*.txt'
    input_txt_filepath_test = os.getcwd(
    ) + f'/../../data/documents/test/*activity_subseq*.txt'

    # _for cluster embeddings
    embeddings_filepath = os.getcwd(
    ) + f'/../../data/sub_sequence_output/word_embeddings_from_clusters.txt'

    vocab, embeddings, corpus, activity_labels, activity_doc_count_index = get_cluster_embeddings(
        input_txt_filepath_train, input_txt_filepath_test, embeddings_filepath)

    num_topics = len(set(activity_labels))
    output_dir = "saved_model"

    # Prepare a trainer
    trainer = GaussianLDAAliasTrainer(
        corpus, embeddings, vocab, num_topics, 0.01, save_path=output_dir, show_topics=num_topics
    )
    # Set training running
    trainer.sample(20)

    activity_topic_mapping = get_activity_topic_mapping(
        list(set(activity_labels)), activity_doc_count_index)

    output_dir = "saved_model"
    model = GaussianLDA.load(output_dir)

    test_docs, test_doc_labels = get_test_documents()

    iterations = 20

    test_results = {}

    test_doc_true = []
    test_doc_glda = []

    for doc, activity in zip(test_docs, test_doc_labels):
        test_topics = model.sample(doc, iterations)

        true_doc_id = int((activity_topic_mapping[activity])[5:])
        test_doc_true.append(true_doc_id)

        test_doc_glda.append(get_mode(test_topics))
        test_results[activity] = (Counter(test_topics), len(test_topics))

    classification_report_dict = classification_report(
        test_doc_true, test_doc_glda, output_dict=True)
    classification_report_dict['adjusted_rand_index_score'] = adjusted_rand_score(
        test_doc_true, test_doc_glda)

    print_testresults(test_results, classification_report_dict)
