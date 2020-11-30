from gaussianlda.model import GaussianLDA

output_dir = "saved_model"
model = GaussianLDA.load(output_dir)

doc = ["sheep", "cow", "bank", "flibble", "sheep", "pig","finance"]
iterations = 100
topics = model.sample(doc, iterations)
print(topics)
