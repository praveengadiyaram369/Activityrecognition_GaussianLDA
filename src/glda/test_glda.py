from gaussianlda.model import GaussianLDA

output_dir = "saved_model"
model = GaussianLDA.load(output_dir)
print(model.log_multivariate_tdensity_tables)
doc = ["sheep", "cow", "bank", "flibble", "sheep", "pig","finance"]
iterations = 100
topics = model.sample(doc, iterations)
print(topics)
