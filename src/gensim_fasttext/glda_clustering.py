import os
import numpy as np
from gaussianlda import GaussianLDAAliasTrainer

from embeddings_generator import get_word_embeddings

if __name__ == "__main__":

    input_txt_filepath = os.getcwd() + f'/../../data/sub_sequence_output/*.txt'
    vocab, embeddings, corpus, labels = get_word_embeddings(input_txt_filepath)

    num_topics = len(set(labels))
    output_dir = "saved_model"

    # Prepare a trainer
    trainer = GaussianLDAAliasTrainer(
        corpus, embeddings, vocab, num_topics, 0.2, save_path=output_dir, show_topics=num_topics
    )
    # Set training running
    trainer.sample(10)