import os
import csv
import numpy as np
from gaussianlda import GaussianLDAAliasTrainer
from gaussianlda.model import GaussianLDA
from collections import Counter

from embeddings_generator import get_cluster_embeddings, get_test_documents
from glda_mapping import get_activity_topic_mapping

from sklearn.metrics import classification_report
from sklearn.metrics.cluster import adjusted_rand_score


def write_testresults(logger, test_results, classification_report_dict, clustering_algo, clustering_cnts, alpha):

    accuracy = classification_report_dict['accuracy']*100
    ari = classification_report_dict['adjusted_rand_index_score']
    weighted_average_precision = classification_report_dict['weighted avg']['precision'] * 100
    weighted_average_recall = classification_report_dict['weighted avg']['recall'] * 100
    weighted_average_f1_score = classification_report_dict['weighted avg']['f1-score'] * 100

    glda_output = [clustering_algo, clustering_cnts, alpha, accuracy, ari,
                   weighted_average_precision, weighted_average_recall, weighted_average_f1_score]

    with open("output/glda_performance_data.csv", "a", newline='') as fp:
        wr = csv.writer(fp, dialect='excel')
        wr.writerow(glda_output)

    logger.info(
        f'Finished writing glda performance metrics to the output')


def test_glda_clustering(logger, activity_topic_mapping, clustering_algo, clustering_cnts, alpha):

    output_dir = "saved_model"
    model = GaussianLDA.load(output_dir)

    test_docs, test_doc_labels = get_test_documents()

    iterations = 150

    test_results = {}

    test_doc_true = []
    test_doc_glda = []

    for doc, activity in zip(test_docs, test_doc_labels):
        test_topics = model.sample(doc, iterations)

        true_doc_id = int((activity_topic_mapping[activity])[5:])
        test_doc_true.extend([true_doc_id] * len(test_topics))

        test_doc_glda.extend(test_topics)
        test_results[activity] = (Counter(test_topics), len(test_topics))

    classification_report_dict = classification_report(
        test_doc_true, test_doc_glda, output_dict=True)
    classification_report_dict['adjusted_rand_index_score'] = adjusted_rand_score(
        test_doc_true, test_doc_glda)

    logger.info(
        f'Finished glda evaluation')

    write_testresults(logger, test_results, classification_report_dict,
                      clustering_algo, clustering_cnts, alpha)


def perform_glda_clustering(logger, clustering_algo, clustering_cnts):

    # _for documents
    input_txt_filepath = os.getcwd() + f'/clustering_output/*activity_subseq_*.txt'

    # _for cluster embeddings
    embeddings_filepath = os.getcwd(
    ) + f'/clustering_output/featues_from_clustering.txt'

    vocab, embeddings, corpus, activity_labels = get_cluster_embeddings(
        input_txt_filepath, embeddings_filepath, clustering_cnts)

    num_topics = len(set(activity_labels))
    output_dir = "saved_model"

    testing_alpha_values = [0.02, 0.1, 0.5]

    for alpha in testing_alpha_values:

        # Prepare a trainer
        trainer = GaussianLDAAliasTrainer(
            corpus, embeddings, vocab, num_topics, alpha, save_path=output_dir, show_topics=num_topics
        )
        # Set training running
        trainer.sample(3)

        logger.info(
            f'Finished glda clustering')

        activity_topic_mapping = get_activity_topic_mapping(activity_labels)

        test_glda_clustering(logger, activity_topic_mapping,
                             clustering_algo, clustering_cnts, alpha)
