import numpy as np
from gaussianlda import GaussianLDAAliasTrainer

# A small vocabulary as a list of words
vocab = "money business bank finance sheep cow goat pig".split()
# A random embedding for each word
# Really, you'd want to load something more useful!
#embeddings = np.random.random_sample((8, 1))

embeddings = np.array([[1,0],[1,0],[1,0],[1,0],[0,1],[0,1],[0,1],[0,1]])

corpus = [
    [0, 2, 1, 1, 3, 0, 6, 1],
    [3, 1, 1, 3, 7, 0, 1, 2],
    [7, 5, 4, 7, 7, 4, 6],
    [5, 6, 1, 7, 7, 5, 6, 4],
]
output_dir = "saved_model"
# Prepare a trainer
trainer = GaussianLDAAliasTrainer(
    corpus, embeddings, vocab, 2, 0.1, 0.1, save_path=output_dir, show_topics=2
)
# Set training running
trainer.sample(100)