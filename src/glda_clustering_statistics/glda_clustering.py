import os
import numpy as np
from gaussianlda import GaussianLDAAliasTrainer
from gaussianlda.model import GaussianLDA
from collections import Counter

from embeddings_generator import get_cluster_embeddings, get_test_documents


def print_testresults(test_results):

    for key, val in test_results.items():
        print(f'{key} -> {val}')


if __name__ == "__main__":

    # _for documents
    input_txt_filepath = os.getcwd() + f'/../../data/documents/*activity_subseq_101*.txt'

    # _for cluster embeddings
    embeddings_filepath = os.getcwd(
    ) + f'/../../data/sub_sequence_output/word_embeddings_from_clusters.txt'

    vocab, embeddings, corpus, labels = get_cluster_embeddings(
        input_txt_filepath, embeddings_filepath)

    num_topics = len(set(labels))
    output_dir = "saved_model"

    # Prepare a trainer
    trainer = GaussianLDAAliasTrainer(
        corpus, embeddings, vocab, num_topics, 0.2, save_path=output_dir, show_topics=num_topics
    )
    # Set training running
    trainer.sample(3)

    output_dir = "saved_model"
    model = GaussianLDA.load(output_dir)

    test_docs, test_doc_labels = get_test_documents()

    iterations = 10

    test_results = {}
    for doc, activity in zip(test_docs, test_doc_labels):
        test_topics = model.sample(doc, iterations)
        test_results[activity] = Counter(test_topics)

    print_testresults(test_results)
